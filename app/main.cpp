// Taver console demo. Links the Taver shared library and exercises both the
// offline (WAV) and live (microphone) scoring paths.
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include "taver/taver.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace {

// A small built-in lexicon in the model's eSpeak-IPA symbols, so the demo can be
// driven with plain words. For arbitrary words, pass --target directly (or use
// tools/g2p.py). These are canonical British/General pronunciations.
const std::unordered_map<std::string, std::string>& lexicon() {
  static const std::unordered_map<std::string, std::string> m = {
      {"tree", "t \xC9\xB9 i\xCB\x90"},          // t ɹ iː
      {"three", "\xCE\xB8 \xC9\xB9 i\xCB\x90"},  // θ ɹ iː
      {"think", "\xCE\xB8 \xC9\xAA \xC5\x8B k"}, // θ ɪ ŋ k
      {"sink", "s \xC9\xAA \xC5\x8B k"},         // s ɪ ŋ k
      {"ship", "\xCA\x83 \xC9\xAA p"},           // ʃ ɪ p
      {"sip", "s \xC9\xAA p"},                   // s ɪ p
      {"right", "\xC9\xB9 a\xC9\xAA t"},         // ɹ aɪ t
      {"light", "l a\xC9\xAA t"},                // l aɪ t
      {"thanks", "\xCE\xB8 \xC3\xA6 \xC5\x8B k s"},
      {"water", "w \xC9\x94 t \xC9\x9A"},        // w ɔ t ɚ
      {"hello", "h \xC9\x99 l o\xCA\x8A"},       // h ə l oʊ
  };
  return m;
}

const char* err_label(int e) {
  switch (e) {
    case TAVER_PHONE_GOOD: return "GOOD";
    case TAVER_PHONE_FAIR: return "fair";
    case TAVER_PHONE_SUBSTITUTED: return "WRONG SOUND";
    case TAVER_PHONE_DELETED: return "SWALLOWED";
    case TAVER_PHONE_INSERTED: return "inserted";
    default: return "?";
  }
}

void print_result(const TaverResult* r, const char* title) {
  std::printf("\n==== %s ====\n", title);
  std::printf("Overall pronunciation score: %.1f / 100   (%.2fs speech, %.1f ms latency)\n",
              r->overall_score, r->utterance_sec, r->latency_ms);
  if (r->num_words > 0) {
    std::printf("  words: ");
    for (int w = 0; w < r->num_words; ++w)
      std::printf("%s(%.0f)%s ", r->words[w].text, r->words[w].score,
                  r->words[w].has_error ? "!" : "");
    std::printf("\n");
  }
  std::printf("  %-10s %-10s %7s   %-12s %s\n", "expected", "you-said", "score", "verdict", "@time");
  std::printf("  ------------------------------------------------------------\n");
  for (int i = 0; i < r->num_phonemes; ++i) {
    const TaverPhonemeScore& p = r->phonemes[i];
    std::printf("  /%-8s/ /%-8s/ %6.1f   %-12s %.2fs\n", p.phoneme, p.realized,
                p.score, err_label(p.error_type), p.start_sec);
  }
  std::printf("  prosody: rate=%.1f ph/s  pitch=%.0f Hz (range %.0f)  fluency=%.0f/100\n",
              r->speaking_rate_pps, r->mean_pitch_hz, r->pitch_range_hz, r->fluency_score);
  std::fflush(stdout);
}

void on_result(const TaverResult* r, void* /*user*/) {
  print_result(r, "Live utterance");
}

void print_metrics(TaverEngine* eng) {
  TaverMetrics m{};
  taver_get_metrics(eng, &m);
  std::printf("\n---- resource & performance metrics ----\n");
  std::printf("  utterances scored : %llu\n", (unsigned long long)m.utterances_scored);
  std::printf("  inference avg     : %.1f ms\n", m.inference_ms_avg);
  std::printf("  e2e latency p50   : %.1f ms\n", m.e2e_latency_ms_p50);
  std::printf("  e2e latency p95   : %.1f ms\n", m.e2e_latency_ms_p95);
  std::printf("  e2e latency max   : %.1f ms\n", m.e2e_latency_ms_max);
  std::printf("  audio chunks lost : %llu\n", (unsigned long long)m.audio_chunks_dropped);
  std::printf("  pool blocks in use: %zu\n", m.pool_blocks_in_use);
  std::printf("  peak working set  : %.1f MB\n", m.peak_working_set_bytes / 1.0e6);
  std::fflush(stdout);
}

void usage() {
  std::printf(
      "Taver - on-device pronunciation assessment\n"
      "Usage:\n"
      "  taver_cli --model M --vocab V\n"
      "              (--text \"any words or a sentence\" | --word W | --target \"p1 p2 ..\")\n"
      "              (--wav FILE | --mic) [--espeak PATH] [--backend auto|cpu|cuda] [--features]\n");
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
#endif

  std::string model = "models/phoneme.onnx";  // FP32: lowest latency on CPU
  std::string vocab = "models/phoneme_vocab.txt";
  std::string espeak = "third_party/espeak/eSpeak NG/espeak-ng.exe";
  std::string word, target, text, wav;
  bool mic = false, features = false;
  int backend = TAVER_BACKEND_AUTO;
  int loop = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* def) -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
    };
    if (a == "--model") model = next(model.c_str());
    else if (a == "--vocab") vocab = next(vocab.c_str());
    else if (a == "--espeak") espeak = next(espeak.c_str());
    else if (a == "--word") word = next("");
    else if (a == "--target") target = next("");
    else if (a == "--text") text = next("");
    else if (a == "--wav") wav = next("");
    else if (a == "--mic") mic = true;
    else if (a == "--loop") loop = std::atoi(next("1").c_str());
    else if (a == "--features") features = true;
    else if (a == "--backend") {
      std::string b = next("auto");
      backend = (b == "cpu") ? TAVER_BACKEND_CPU
              : (b == "cuda") ? TAVER_BACKEND_CUDA
              : (b == "dml" || b == "directml") ? TAVER_BACKEND_DIRECTML
              : TAVER_BACKEND_AUTO;
    } else if (a == "-h" || a == "--help") { usage(); return 0; }
  }

  if (target.empty() && text.empty() && !word.empty()) {
    auto it = lexicon().find(word);
    if (it != lexicon().end()) target = it->second;
    else text = word;  // fall through to G2P for words outside the demo lexicon
  }
  if (target.empty() && text.empty()) { usage(); return 2; }
  if (wav.empty() && !mic) { usage(); return 2; }

  std::printf("%s | model: %s\n", taver_version(), model.c_str());

  TaverConfig cfg{};
  cfg.model_path = model.c_str();
  cfg.vocab_path = vocab.c_str();
  cfg.espeak_path = espeak.c_str();
  cfg.lexicon_path = "data/lexicon.txt";
  cfg.calibration_path = "config/calibration.json";
  cfg.backend = backend;
  cfg.g2p_voice = 0;
  cfg.print_features = features ? 1 : 0;
  cfg.on_result = on_result;
  cfg.user_data = nullptr;

  TaverEngine* eng = nullptr;
  TaverStatus st = taver_create(&cfg, &eng);
  if (st != TAVER_OK) {
    std::printf("create failed: %s\n", taver_status_string(st));
    return 1;
  }

  if (!text.empty()) {
    st = taver_set_target_text(eng, text.c_str());
  } else {
    st = taver_set_target(eng, target.c_str());
  }
  if (st != TAVER_OK) {
    std::printf("set_target failed: %s\n", taver_status_string(st));
    taver_destroy(eng);
    return 1;
  }
  {
    char ph[1024] = {0}, ipa[1024] = {0};
    if (taver_get_target_info(eng, ph, sizeof(ph), ipa, sizeof(ipa)) == TAVER_OK) {
      if (!text.empty()) std::printf("Text: \"%s\"  ->  IPA: %s\n", text.c_str(), ipa);
      std::printf("Target phonemes: %s\n", ph);
    }
  }

  if (!wav.empty()) {
    TaverResult r{};
    if (loop > 1) {
      // Stress / leak check: score the same file many times in one process and
      // watch the working set. Flat memory == the pre-allocated pools work.
      std::printf("\nStress test: %d iterations (watch the working set stay flat)...\n", loop);
      for (int i = 0; i < loop; ++i) {
        st = taver_score_wav(eng, wav.c_str(), &r);
        if (st != TAVER_OK) { std::printf("iter %d failed: %s\n", i, taver_status_string(st)); break; }
        if (i == 0 || (i + 1) % 25 == 0 || i == loop - 1) {
          TaverMetrics m{};
          taver_get_metrics(eng, &m);
          std::printf("  iter %4d: score=%.1f  peakWS=%.1f MB  poolInUse=%zu  dropped=%llu\n",
                      i + 1, r.overall_score, m.peak_working_set_bytes / 1.0e6,
                      m.pool_blocks_in_use, (unsigned long long)m.audio_chunks_dropped);
        }
      }
    } else {
      st = taver_score_wav(eng, wav.c_str(), &r);
      if (st != TAVER_OK) {
        std::printf("score_wav failed: %s\n", taver_status_string(st));
        taver_destroy(eng);
        return 1;
      }
      print_result(&r, wav.c_str());
    }
  } else {
    std::printf("\nListening on the default microphone. Speak the word, then press Enter to stop...\n");
    st = taver_start(eng);
    if (st != TAVER_OK) {
      std::printf("start failed: %s\n", taver_status_string(st));
      taver_destroy(eng);
      return 1;
    }
    std::getchar();
    taver_stop(eng);
  }

  print_metrics(eng);
  taver_destroy(eng);
  return 0;
}
