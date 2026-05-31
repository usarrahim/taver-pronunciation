// The asynchronous pipeline that ties the four layers together.
//
//   [audio source] --chunks--> (SPSC) --> [DSP+VAD thread] --utterances-->
//        (SPSC of pooled Utterance*) --> [inference+scoring thread] --> callback
//
// Audio capture, feature extraction, and neural inference run on independent
// threads communicating through lock-free SPSC rings and a fixed object pool,
// so no stage ever blocks another and the steady-state allocation count is zero.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "taver/taver.h"
#include "taver/spsc_ring.h"
#include "taver/memory_pool.h"
#include "taver/types.h"
#include "audio/audio_source.h"
#include "dsp/feature_extractor.h"
#include "ai/onnx_inference_engine.h"
#include "align/vocabulary.h"
#include "align/gop_scorer.h"
#include "g2p/g2p.h"

namespace taver {

class Engine {
 public:
  Engine();
  ~Engine();

  TaverStatus init(const TaverConfig& cfg);
  TaverStatus set_target(const std::string& phoneme_string);
  TaverStatus set_target_text(const std::string& text);
  void set_g2p_voice(int voice);  // 0=en-us, 1=en-gb
  bool target_info(std::string& phonemes, std::string& ipa);

  TaverStatus start_microphone();
  void stop();

  TaverStatus score_wav(const std::string& path, TaverResult* out);
  // Score an in-memory mono 16 kHz buffer synchronously (one utterance).
  TaverStatus score_samples(const float* samples, int n, TaverResult* out);

  void get_metrics(TaverMetrics* out);

  const std::string& last_error() const { return last_error_; }

 private:
  void dsp_loop();
  void infer_loop();
  void on_audio_chunk(const AudioChunk& chunk);     // producer (audio thread)
  void finalize_utterance();                        // close current utterance
  void build_result(const align::AlignResult& a, int utterance_samples,
                    TaverResult* out);
  void aggregate_words(TaverResult* out);
  void record_latency(double ms);

  // Config / shared models
  TaverConfig cfg_{};
  bool print_features_ = false;
  align::Vocabulary vocab_;
  ai::OnnxInferenceEngine model_;
  std::unique_ptr<align::GopScorer> scorer_;
  g2p::G2P g2p_;
  std::vector<int> target_ids_;
  std::vector<int> target_word_start_;        // phoneme idx where each word starts
  std::vector<std::string> target_words_;     // orthographic words (text targets)
  std::string last_ipa_;                      // raw IPA of last text target
  std::string last_phonemes_;                 // space-joined phoneme symbols
  std::mutex target_mutex_;
  std::string last_error_;

  // Layer A queue + source
  std::unique_ptr<SpscRing<AudioChunk>> audio_q_;
  std::unique_ptr<audio::IAudioSource> source_;

  // Layer B
  dsp::FeatureExtractor features_;
  std::vector<float> frame_buf_;     // streaming framing buffer
  std::size_t frame_pos_ = 0;
  std::vector<float> preroll_;       // ring of recent raw samples
  std::size_t preroll_pos_ = 0;
  bool preroll_full_ = false;

  // VAD state
  bool in_speech_ = false;
  int active_run_ = 0;
  int hangover_ = 0;

  // Utterance pooling + inference queue
  std::unique_ptr<MemoryPool<Utterance>> utt_pool_;
  std::unique_ptr<MemoryPool<LogitMatrix>> logit_pool_;
  std::unique_ptr<SpscRing<Utterance*>> infer_q_;
  Utterance* current_utt_ = nullptr;

  // Threads / lifecycle
  std::thread dsp_thread_;
  std::thread infer_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> file_mode_{false};
  std::atomic<bool> producer_done_{false};  // file source emitted all audio
  std::atomic<bool> dsp_done_{false};        // DSP drained + flushed
  std::atomic<bool> infer_busy_{false};
  float sync_best_sec_ = -1.0f;              // pick longest utterance in a file

  // Synchronous (score_wav) result handoff
  std::mutex sync_mutex_;
  std::condition_variable sync_cv_;
  TaverResult* sync_out_ = nullptr;
  bool sync_have_result_ = false;

  // Metrics
  std::mutex metrics_mutex_;
  std::vector<double> latencies_ms_;
  double inference_ms_sum_ = 0.0;
  std::atomic<std::uint64_t> utt_scored_{0};
  std::atomic<std::uint64_t> chunks_dropped_{0};
};

}  // namespace taver
