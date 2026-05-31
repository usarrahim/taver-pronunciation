// Converts a single 25 ms analysis frame of 16 kHz audio into log-mel + MFCC
// features. Filterbank, window and DCT basis are precomputed; compute() touches
// only preallocated scratch and writes into a caller-owned FeatureFrame.
#pragma once

#include <array>
#include <vector>

#include "taver/config.h"
#include "taver/types.h"
#include "dsp/fft.h"

namespace taver::dsp {

class FeatureExtractor {
 public:
  FeatureExtractor();

  // `frame` must contain kFrameLengthSamples mono samples in [-1, 1].
  void compute(const float* frame, FeatureFrame& out);

 private:
  Fft fft_;
  std::array<float, kFrameLengthSamples> window_{};   // Hann
  std::vector<float> mel_weights_;                     // kNumMelBins * num_bins
  std::array<int, kNumMelBins> mel_start_{};           // first bin per filter
  std::array<int, kNumMelBins> mel_end_{};             // one past last bin
  std::vector<float> dct_;                             // kNumMfcc * kNumMelBins
  int num_bins_;                                       // kFftSize/2 + 1

  // Scratch (reused every call, never reallocated):
  std::array<float, kFftSize> windowed_{};
  std::array<float, kFftSize> re_{};
  std::array<float, kFftSize> im_{};
  std::vector<float> power_;
};

}  // namespace taver::dsp
