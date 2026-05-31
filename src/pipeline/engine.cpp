#include "pipeline/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>

#include "audio/wasapi_source.h"
#include "audio/wav_file_source.h"
#include "prosody/prosody.h"
#include "g2p/lexicon.h"

namespace taver {

namespace {
constexpr int kPrerollSamples = 1920;   // 120 ms of pre-onset context
constexpr int kAudioQueueChunks = 256;  // ~2.5 s of 10 ms chunks
constexpr int kUttPoolSize = 4;
constexpr int kLogitPoolSize = 4;
constexpr std::size_t kLatencyWindow = 256;

std::uint64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

void sleep_us(int us) {
  std::this_thread::sleep_for(std::chrono::microseconds(us));
}
}  // namespace

Engine::Engine() = default;

Engine::~Engine() { stop(); }

TaverStatus Engine::init(const TaverConfig& cfg) {
  cfg_ = cfg;
  print_features_ = cfg.print_features != 0;

  if (!cfg.model_path || !cfg.vocab_path) {
    last_error_ = "model_path and vocab_path are required";
    return TAVER_ERR_INVALID_ARG;
  }

  std::string err;
  if (!vocab_.load(cfg.vocab_path, err)) {
    last_error_ = err;
    return TAVER_ERR_MODEL_LOAD;
  }

  ai::Backend backend = ai::Backend::kAuto;
  if (cfg.backend == TAVER_BACKEND_CPU) backend = ai::Backend::kCpu;
  else if (cfg.backend == TAVER_BACKEND_CUDA) backend = ai::Backend::kCuda;
  else if (cfg.backend == TAVER_BACKEND_DIRECTML) backend = ai::Backend::kDirectML;

  if (!model_.load(cfg.model_path, backend, err)) {
    last_error_ = err;
    return TAVER_ERR_MODEL_LOAD;
  }

  scorer_ = std::make_unique<align::GopScorer>(vocab_);
  if (cfg.calibration_path && cfg.calibration_path[0]) {
    align::GopCalibration cal;
    if (align::load_calibration(cfg.calibration_path, cal)) scorer_->set_calibration(cal);
  }

  static g2p::Lexicon lex;
  if (cfg.lexicon_path && cfg.lexicon_path[0]) {
    std::string lerr;
    if (!lex.load(cfg.lexicon_path, lerr))
      std::fprintf(stderr, "[Taver] lexicon: %s\n", lerr.c_str());
  }

  if (cfg.espeak_path && cfg.espeak_path[0]) {
    std::string gerr;
    if (!g2p_.init(cfg.espeak_path, vocab_, gerr))
      std::fprintf(stderr, "[Taver] G2P disabled: %s\n", gerr.c_str());
    else {
      g2p_.set_voice(cfg.g2p_voice);
      if (!lex.empty()) g2p_.set_lexicon(&lex);
    }
  }

  audio_q_ = std::make_unique<SpscRing<AudioChunk>>(kAudioQueueChunks);
  infer_q_ = std::make_unique<SpscRing<Utterance*>>(kUttPoolSize + 2);
  utt_pool_ = std::make_unique<MemoryPool<Utterance>>(kUttPoolSize);
  logit_pool_ = std::make_unique<MemoryPool<LogitMatrix>>(kLogitPoolSize);

  frame_buf_.reserve(kFrameLengthSamples * 4);
  preroll_.assign(kPrerollSamples, 0.0f);
  latencies_ms_.reserve(kLatencyWindow);
  return TAVER_OK;
}

TaverStatus Engine::set_target(const std::string& phoneme_string) {
  std::vector<int> ids;
  std::istringstream ss(phoneme_string);
  std::string tok;
  std::string missing;
  while (ss >> tok) {
    const int id = vocab_.id(tok);
    if (id < 0) {
      if (!missing.empty()) missing += ", ";
      missing += tok;
    } else {
      ids.push_back(id);
    }
  }
  if (ids.empty()) {
    last_error_ = "target produced no known phonemes (unknown: " + missing + ")";
    return TAVER_ERR_INVALID_ARG;
  }
  if (!missing.empty()) {
    last_error_ = "ignored unknown phoneme symbols: " + missing;
  }
  if (static_cast<int>(ids.size()) > kMaxTargetPhonemes) {
    last_error_ = "target sequence too long";
    return TAVER_ERR_INVALID_ARG;
  }
  std::string joined;
  for (int id : ids) { if (!joined.empty()) joined += ' '; joined += vocab_.symbol(id); }
  std::lock_guard<std::mutex> lk(target_mutex_);
  target_ids_ = std::move(ids);
  target_word_start_.clear();   // raw phoneme target: no word grouping
  target_words_.clear();
  last_phonemes_ = joined;
  last_ipa_.clear();
  return TAVER_OK;
}

void Engine::set_g2p_voice(int voice) { g2p_.set_voice(voice); }

TaverStatus Engine::set_target_text(const std::string& text) {
  if (!g2p_.available()) {
    last_error_ = "text targets need espeak (set espeak_path in config)";
    return TAVER_ERR_INVALID_ARG;
  }
  g2p::G2PResult g;
  std::string gerr;
  if (!g2p_.convert(text, g, gerr)) { last_error_ = gerr; return TAVER_ERR_INVALID_ARG; }
  if (static_cast<int>(g.phoneme_ids.size()) > kMaxTargetPhonemes) {
    last_error_ = "phrase too long (" + std::to_string(g.phoneme_ids.size()) +
                  " phonemes, max " + std::to_string(kMaxTargetPhonemes) + ")";
    return TAVER_ERR_INVALID_ARG;
  }
  std::string joined;
  for (int id : g.phoneme_ids) { if (!joined.empty()) joined += ' '; joined += vocab_.symbol(id); }

  std::lock_guard<std::mutex> lk(target_mutex_);
  target_ids_ = std::move(g.phoneme_ids);
  target_word_start_ = std::move(g.word_start);
  target_words_ = std::move(g.words);
  last_phonemes_ = joined;
  last_ipa_ = g.ipa;
  return TAVER_OK;
}

bool Engine::target_info(std::string& phonemes, std::string& ipa) {
  std::lock_guard<std::mutex> lk(target_mutex_);
  if (target_ids_.empty()) return false;
  phonemes = last_phonemes_;
  ipa = last_ipa_;
  return true;
}

// ---- producer: audio thread -> SPSC ----------------------------------------
void Engine::on_audio_chunk(const AudioChunk& chunk) {
  if (!audio_q_->push(chunk)) {
    chunks_dropped_.fetch_add(1, std::memory_order_relaxed);
  }
}

void Engine::finalize_utterance() {
  if (!current_utt_) return;
  // Trim most of the trailing VAD hang-over silence so the acoustic model only
  // sees speech (+ a small margin). This cuts both inference time and latency.
  if (hangover_ >= kVadHangoverFrames) {
    const std::size_t trim =
        static_cast<std::size_t>(hangover_ - kKeepTrailingFrames) * kCaptureChunkSamples;
    if (current_utt_->length > trim) current_utt_->length -= trim;
  }
  current_utt_->end_ns = now_ns();
  if (current_utt_->length > 0 && infer_q_->push(current_utt_)) {
    // ownership transferred to the inference thread
  } else {
    utt_pool_->release(current_utt_);  // dropped (too short or queue full)
  }
  current_utt_ = nullptr;
  in_speech_ = false;
  hangover_ = 0;
  active_run_ = 0;
}

void Engine::dsp_loop() {
  AudioChunk chunk;
  std::int64_t frame_index = 0;

  while (running_.load()) {
    if (!audio_q_->pop(chunk)) {
      if (file_mode_.load() && producer_done_.load()) {
        finalize_utterance();  // flush trailing speech at end of file
        dsp_done_.store(true);
        return;
      }
      sleep_us(500);
      continue;
    }

    const std::uint32_t n = chunk.count;
    if (n == 0) continue;

    // VAD energy for this 10 ms chunk.
    float e = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) e += chunk.samples[i] * chunk.samples[i];
    e /= n;

    // Maintain pre-roll ring (raw samples) so onset is never clipped.
    for (std::uint32_t i = 0; i < n; ++i) {
      preroll_[preroll_pos_] = chunk.samples[i];
      preroll_pos_ = (preroll_pos_ + 1) % preroll_.size();
      if (preroll_pos_ == 0) preroll_full_ = true;
    }

    // While in speech, append raw samples to the active utterance.
    if (in_speech_ && current_utt_) {
      const std::size_t cap = current_utt_->samples.size();
      for (std::uint32_t i = 0; i < n && current_utt_->length < cap; ++i)
        current_utt_->samples[current_utt_->length++] = chunk.samples[i];
    }

    // Feature extraction (Phase 1 diagnostics) on overlapping frames.
    for (std::uint32_t i = 0; i < n; ++i) frame_buf_.push_back(chunk.samples[i]);
    while (frame_buf_.size() - frame_pos_ >= static_cast<std::size_t>(kFrameLengthSamples)) {
      FeatureFrame ff;
      features_.compute(frame_buf_.data() + frame_pos_, ff);
      ff.voiced = in_speech_;
      if (print_features_ && (frame_index % 4 == 0)) {
        std::fprintf(stderr, "[t=%6.2fs] E=%.5f V=%d mfcc:", frame_index * 0.01,
                     ff.energy, ff.voiced ? 1 : 0);
        for (int c = 0; c < 8 && c < kNumMfcc; ++c)
          std::fprintf(stderr, " %+6.2f", ff.mfcc[c]);
        std::fprintf(stderr, " ...\n");
      }
      frame_pos_ += kFrameShiftSamples;
      ++frame_index;
    }
    if (frame_pos_ > static_cast<std::size_t>(kFrameLengthSamples) * 8) {
      frame_buf_.erase(frame_buf_.begin(), frame_buf_.begin() + frame_pos_);
      frame_pos_ = 0;
    }

    // VAD state machine.
    if (!in_speech_) {
      active_run_ = (e > kVadOnsetEnergy) ? active_run_ + 1 : 0;
      if (active_run_ >= kVadMinSpeechFrames) {
        current_utt_ = utt_pool_->acquire();
        if (current_utt_) {
          current_utt_->reset();
          // seed with pre-roll (oldest -> newest)
          const std::size_t sz = preroll_.size();
          const std::size_t start = preroll_full_ ? preroll_pos_ : 0;
          const std::size_t count = preroll_full_ ? sz : preroll_pos_;
          for (std::size_t j = 0; j < count; ++j)
            current_utt_->samples[current_utt_->length++] = preroll_[(start + j) % sz];
          current_utt_->start_ns = chunk.timestamp_ns;
          in_speech_ = true;
          hangover_ = 0;
        } else {
          active_run_ = 0;  // no pool block free; keep waiting
        }
      }
    } else {
      hangover_ = (e < kVadOffsetEnergy) ? hangover_ + 1 : 0;
      const bool too_long =
          current_utt_ && current_utt_->length + kCaptureChunkSamples >=
                              current_utt_->samples.size();
      if (hangover_ >= kVadHangoverFrames || too_long) finalize_utterance();
    }
  }
}

void Engine::infer_loop() {
  while (running_.load()) {
    Utterance* utt = nullptr;
    if (!infer_q_->pop(utt)) {
      if (file_mode_.load() && dsp_done_.load()) return;  // fully drained
      sleep_us(500);
      continue;
    }
    infer_busy_.store(true);

    LogitMatrix* logit = logit_pool_->acquire();
    if (!logit) { utt_pool_->release(utt); infer_busy_.store(false); continue; }

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = model_.infer(utt->samples.data(),
                                 static_cast<int>(utt->length), *logit);
    const double infer_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

    if (ok) {
      std::vector<int> target;
      { std::lock_guard<std::mutex> lk(target_mutex_); target = target_ids_; }

      align::AlignResult ar;
      if (!target.empty() &&
          scorer_->score(*logit, target, static_cast<int>(utt->length), ar)) {
        TaverResult result{};
        const std::uint64_t now = now_ns();
        build_result(ar, static_cast<int>(utt->length), &result);
        prosody::ProsodyStats pr;
        prosody::analyze(utt->samples.data(), static_cast<int>(utt->length),
                         kTargetSampleRate, result.num_phonemes, pr);
        result.speaking_rate_pps = pr.speaking_rate_pps;
        result.mean_pitch_hz = pr.mean_pitch_hz;
        result.pitch_range_hz = pr.pitch_range_hz;
        result.fluency_score = pr.fluency_score;
        result.latency_ms = static_cast<float>((now - utt->end_ns) / 1.0e6);

        {
          std::lock_guard<std::mutex> lk(metrics_mutex_);
          inference_ms_sum_ += infer_ms;
        }
        record_latency(result.latency_ms);
        utt_scored_.fetch_add(1, std::memory_order_relaxed);

        if (file_mode_.load()) {
          std::lock_guard<std::mutex> lk(sync_mutex_);
          if (sync_out_ && result.utterance_sec > sync_best_sec_) {
            *sync_out_ = result;
            sync_best_sec_ = result.utterance_sec;
            sync_have_result_ = true;
          }
        } else if (cfg_.on_result) {
          cfg_.on_result(&result, cfg_.user_data);
        }
      }
    }

    logit_pool_->release(logit);
    utt_pool_->release(utt);
    infer_busy_.store(false);
  }
}

void Engine::build_result(const align::AlignResult& a, int utterance_samples,
                          TaverResult* out) {
  out->overall_score = a.overall_score;
  out->utterance_sec = static_cast<float>(utterance_samples) / kTargetSampleRate;
  out->num_phonemes = std::min<int>(static_cast<int>(a.phones.size()), TAVER_MAX_PHONEMES);
  for (int i = 0; i < out->num_phonemes; ++i) {
    const auto& p = a.phones[i];
    TaverPhonemeScore& d = out->phonemes[i];
    std::snprintf(d.phoneme, TAVER_PHONEME_LEN, "%s", p.symbol.c_str());
    std::snprintf(d.realized, TAVER_PHONEME_LEN, "%s", p.realized.c_str());
    d.score = p.score;
    d.error_type = static_cast<int>(p.error);
    d.start_sec = p.start_sec;
    d.duration_sec = p.duration_sec;
  }
  aggregate_words(out);
}

// Collapse per-phoneme scores into per-word scores using the target's word
// boundaries (only present when the target was set from text).
void Engine::aggregate_words(TaverResult* out) {
  std::vector<int> starts;
  std::vector<std::string> words;
  { std::lock_guard<std::mutex> lk(target_mutex_);
    starts = target_word_start_; words = target_words_; }

  out->num_words = 0;
  if (starts.empty() || words.empty()) return;

  // starts[] and words[] are kept the same length by the G2P layer; clamp to the
  // smaller of the two (and the result capacity) so we never index past either.
  const int nw = std::min<int>({static_cast<int>(words.size()),
                                static_cast<int>(starts.size()), TAVER_MAX_WORDS});
  for (int w = 0; w < nw; ++w) {
    const int p0 = starts[w];
    const int p1 = (w + 1 < static_cast<int>(starts.size())) ? starts[w + 1]
                                                             : out->num_phonemes;
    if (p0 >= out->num_phonemes) break;
    const int hi = std::min(p1, out->num_phonemes);
    TaverWordScore& d = out->words[out->num_words];
    std::snprintf(d.text, TAVER_WORD_LEN, "%s", words[w].c_str());
    d.phoneme_start = p0;
    d.phoneme_count = hi - p0;
    float sum = 0.0f; int cnt = 0, err = 0;
    for (int i = p0; i < hi; ++i) {
      sum += out->phonemes[i].score; ++cnt;
      const int et = out->phonemes[i].error_type;
      if (et == TAVER_PHONE_SUBSTITUTED || et == TAVER_PHONE_DELETED) err = 1;
    }
    d.score = cnt ? sum / cnt : 0.0f;
    d.has_error = err;
    ++out->num_words;
  }
}

void Engine::record_latency(double ms) {
  std::lock_guard<std::mutex> lk(metrics_mutex_);
  if (latencies_ms_.size() >= kLatencyWindow)
    latencies_ms_.erase(latencies_ms_.begin());
  latencies_ms_.push_back(ms);
}

TaverStatus Engine::start_microphone() {
  if (running_.load()) return TAVER_OK;
  running_.store(true);
  file_mode_.store(false);
  producer_done_.store(false);
  dsp_done_.store(false);

  dsp_thread_ = std::thread([this] { dsp_loop(); });
  infer_thread_ = std::thread([this] { infer_loop(); });

  source_ = std::make_unique<audio::WasapiSource>();
  if (!source_->start([this](const AudioChunk& c) { on_audio_chunk(c); })) {
    last_error_ = static_cast<audio::WasapiSource*>(source_.get())->last_error();
    stop();
    return TAVER_ERR_AUDIO_DEVICE;
  }
  return TAVER_OK;
}

namespace {
// Trim leading/trailing silence from a mono buffer using short-window energy,
// keeping an 80 ms margin so onsets/offsets are not clipped.
void trim_silence(const float* x, int n, int& start, int& end) {
  start = 0; end = n;
  const int win = 160;  // 10 ms
  const int nf = n / win;
  if (nf < 2) return;
  std::vector<float> e(nf, 0.0f);
  float peak = 0.0f;
  for (int f = 0; f < nf; ++f) {
    float s = 0.0f;
    for (int i = 0; i < win; ++i) { const float v = x[f * win + i]; s += v * v; }
    e[f] = s / win;
    peak = std::max(peak, e[f]);
  }
  const float thr = std::max(1e-5f, peak * 0.02f);
  int first = -1, last = -1;
  for (int f = 0; f < nf; ++f) if (e[f] > thr) { if (first < 0) first = f; last = f; }
  if (first < 0) return;  // all silence: keep whole buffer
  const int margin = 8;   // 80 ms
  start = std::max(0, (first - margin) * win);
  end = std::min(n, (last + 1 + margin) * win);
}
}  // namespace

TaverStatus Engine::score_wav(const std::string& path, TaverResult* out) {
  if (!out) return TAVER_ERR_INVALID_ARG;
  audio::WavFileSource src(path);
  if (!src.ok()) { last_error_ = "cannot open/decode WAV: " + path; return TAVER_ERR_FILE; }
  const auto& mono = src.mono();
  return score_samples(mono.data(), static_cast<int>(mono.size()), out);
}

// Synchronous single-utterance scoring. Used for WAV files, recorded clips from
// the web UI, and the offline evaluation harness. No threads, no VAD splitting:
// the whole clip is treated as one utterance (silence-trimmed), so sentences and
// multi-word phrases align cleanly against the full target.
TaverStatus Engine::score_samples(const float* samples, int n, TaverResult* out) {
  if (!out || !samples) return TAVER_ERR_INVALID_ARG;
  std::vector<int> target;
  { std::lock_guard<std::mutex> lk(target_mutex_); target = target_ids_; }
  if (target.empty()) { last_error_ = "no target set"; return TAVER_ERR_NO_TARGET; }

  *out = TaverResult{};

  int s = 0, e = n;
  trim_silence(samples, n, s, e);
  int len = e - s;
  if (len < kFrameLengthSamples) { last_error_ = "utterance too short"; return TAVER_ERR_RUNTIME; }
  if (len > kMaxUtteranceSamples) len = kMaxUtteranceSamples;

  LogitMatrix* logit = logit_pool_->acquire();
  if (!logit) { last_error_ = "no logit buffer"; return TAVER_ERR_RUNTIME; }

  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = model_.infer(samples + s, len, *logit);
  const double infer_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!ok) { logit_pool_->release(logit); last_error_ = "inference failed"; return TAVER_ERR_RUNTIME; }

  align::AlignResult ar;
  if (!scorer_->score(*logit, target, len, ar)) {
    logit_pool_->release(logit); last_error_ = "alignment failed"; return TAVER_ERR_RUNTIME;
  }
  build_result(ar, len, out);

  prosody::ProsodyStats pr;
  prosody::analyze(samples + s, len, kTargetSampleRate, out->num_phonemes, pr);
  out->speaking_rate_pps = pr.speaking_rate_pps;
  out->mean_pitch_hz = pr.mean_pitch_hz;
  out->pitch_range_hz = pr.pitch_range_hz;
  out->fluency_score = pr.fluency_score;
  out->latency_ms = static_cast<float>(infer_ms);

  logit_pool_->release(logit);
  { std::lock_guard<std::mutex> lk(metrics_mutex_); inference_ms_sum_ += infer_ms; }
  record_latency(out->latency_ms);
  utt_scored_.fetch_add(1, std::memory_order_relaxed);
  return TAVER_OK;
}

void Engine::stop() {
  if (!running_.load() && !dsp_thread_.joinable() && !infer_thread_.joinable())
    return;
  if (source_) source_->stop();
  running_.store(false);
  if (dsp_thread_.joinable()) dsp_thread_.join();
  if (infer_thread_.joinable()) infer_thread_.join();
  source_.reset();
}

void Engine::get_metrics(TaverMetrics* out) {
  if (!out) return;
  *out = TaverMetrics{};
  std::vector<double> snap;
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    snap = latencies_ms_;
    out->inference_ms_avg =
        utt_scored_.load() ? inference_ms_sum_ / utt_scored_.load() : 0.0;
  }
  if (!snap.empty()) {
    std::sort(snap.begin(), snap.end());
    auto pct = [&](double p) {
      const std::size_t idx = std::min(snap.size() - 1,
          static_cast<std::size_t>(p * (snap.size() - 1)));
      return snap[idx];
    };
    out->e2e_latency_ms_p50 = pct(0.50);
    out->e2e_latency_ms_p95 = pct(0.95);
    out->e2e_latency_ms_max = snap.back();
  }
  out->utterances_scored = utt_scored_.load();
  out->audio_chunks_dropped = chunks_dropped_.load();
  out->pool_blocks_in_use =
      utt_pool_ ? (utt_pool_->capacity() - utt_pool_->available()) : 0;
  out->peak_working_set_bytes = 0;  // filled by the C API layer (psapi)
}

}  // namespace taver
