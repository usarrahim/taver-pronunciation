// Plain-old-data structures that flow between pipeline stages. Kept free of
// behaviour so they can be copied across SPSC queues and pooled cheaply.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "Taver/config.h"

namespace taver {

// A small fixed chunk of mono 16 kHz float samples handed from the audio
// callback to the DSP thread through the SPSC ring.
struct AudioChunk {
  std::array<float, kCaptureChunkSamples> samples{};
  std::uint32_t count = 0;          // valid samples in `samples`
  std::uint64_t timestamp_ns = 0;   // capture time of first sample
};

// One frame of acoustic features produced by the DSP stage.
struct FeatureFrame {
  std::array<float, kNumMelBins> log_mel{};
  std::array<float, kNumMfcc>    mfcc{};
  float energy = 0.0f;              // mean-square of the windowed frame
  bool  voiced = false;             // VAD decision for this frame
};

// A complete captured utterance (raw 16 kHz audio) pending inference.
// Large, so it lives in a MemoryPool and is passed by pointer.
struct Utterance {
  std::vector<float> samples;       // capacity == kMaxUtteranceSamples
  std::size_t length = 0;
  std::uint64_t start_ns = 0;       // capture time of speech onset
  std::uint64_t end_ns = 0;         // capture time of speech offset

  Utterance() { samples.resize(kMaxUtteranceSamples); }
  void reset() { length = 0; start_ns = end_ns = 0; }
};

// Frame-level phoneme log-probabilities from the acoustic model.
// Stored row-major: [frame][vocab]. Lives in a MemoryPool.
struct LogitMatrix {
  std::vector<float> data;          // num_frames * vocab_size
  int num_frames = 0;
  int vocab_size = 0;
  std::uint64_t utt_start_ns = 0;
  std::uint64_t utt_end_ns = 0;

  LogitMatrix() { data.resize(static_cast<std::size_t>(kMaxModelFrames) * kMaxVocab); }
  float* row(int f) { return data.data() + static_cast<std::size_t>(f) * vocab_size; }
  const float* row(int f) const {
    return data.data() + static_cast<std::size_t>(f) * vocab_size;
  }
};

}  // namespace taver
