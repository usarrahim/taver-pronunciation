#include "dsp/feature_extractor.h"

#include <cmath>

namespace taver::dsp {

namespace {
inline float hz_to_mel(float hz) {
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}
inline float mel_to_hz(float mel) {
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}
}  // namespace

FeatureExtractor::FeatureExtractor()
    : fft_(kFftSize), num_bins_(kFftSize / 2 + 1) {
  power_.resize(num_bins_);

  // Periodic Hann window (matches torch/librosa default for STFT).
  for (int i = 0; i < kFrameLengthSamples; ++i) {
    window_[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i /
                                        static_cast<float>(kFrameLengthSamples));
  }

  // Triangular mel filterbank over the power-spectrum bins.
  mel_weights_.assign(static_cast<std::size_t>(kNumMelBins) * num_bins_, 0.0f);
  const float mel_lo = hz_to_mel(kMelLowHz);
  const float mel_hi = hz_to_mel(kMelHighHz);
  std::array<float, kNumMelBins + 2> mel_pts{};
  for (int i = 0; i < kNumMelBins + 2; ++i) {
    const float mel = mel_lo + (mel_hi - mel_lo) * i / (kNumMelBins + 1);
    mel_pts[i] = mel_to_hz(mel);
  }
  const float bin_hz = static_cast<float>(kTargetSampleRate) / kFftSize;
  for (int m = 0; m < kNumMelBins; ++m) {
    const float f_left = mel_pts[m];
    const float f_center = mel_pts[m + 1];
    const float f_right = mel_pts[m + 2];
    int start = num_bins_, end = 0;
    for (int k = 0; k < num_bins_; ++k) {
      const float f = k * bin_hz;
      float w = 0.0f;
      if (f >= f_left && f <= f_center) {
        w = (f - f_left) / (f_center - f_left);
      } else if (f > f_center && f <= f_right) {
        w = (f_right - f) / (f_right - f_center);
      }
      if (w > 0.0f) {
        mel_weights_[static_cast<std::size_t>(m) * num_bins_ + k] = w;
        if (k < start) start = k;
        if (k + 1 > end) end = k + 1;
      }
    }
    mel_start_[m] = (start <= end) ? start : 0;
    mel_end_[m] = (start <= end) ? end : 0;
  }

  // Orthonormal DCT-II basis (type 2), as used for MFCC.
  dct_.resize(static_cast<std::size_t>(kNumMfcc) * kNumMelBins);
  for (int k = 0; k < kNumMfcc; ++k) {
    const float scale =
        (k == 0) ? std::sqrt(1.0f / kNumMelBins) : std::sqrt(2.0f / kNumMelBins);
    for (int m = 0; m < kNumMelBins; ++m) {
      dct_[static_cast<std::size_t>(k) * kNumMelBins + m] =
          scale * std::cos(static_cast<float>(M_PI) * k * (2 * m + 1) /
                           (2.0f * kNumMelBins));
    }
  }
}

void FeatureExtractor::compute(const float* frame, FeatureFrame& out) {
  // Frame energy (mean square of raw samples) drives the VAD.
  float energy = 0.0f;
  for (int i = 0; i < kFrameLengthSamples; ++i) energy += frame[i] * frame[i];
  out.energy = energy / kFrameLengthSamples;

  // Pre-emphasis + Hann window into the zero-padded FFT buffer.
  float prev = frame[0];
  for (int i = 0; i < kFftSize; ++i) {
    if (i < kFrameLengthSamples) {
      const float pe = frame[i] - kPreemphasis * prev;
      prev = frame[i];
      windowed_[i] = pe * window_[i];
    } else {
      windowed_[i] = 0.0f;
    }
  }

  fft_.power_spectrum(windowed_.data(), power_.data(), re_.data(), im_.data());

  // Mel projection + log compression.
  for (int m = 0; m < kNumMelBins; ++m) {
    float acc = 0.0f;
    const float* w = &mel_weights_[static_cast<std::size_t>(m) * num_bins_];
    for (int k = mel_start_[m]; k < mel_end_[m]; ++k) acc += w[k] * power_[k];
    out.log_mel[m] = std::log(acc + 1.0e-10f);
  }

  // MFCC via DCT-II of the log-mel spectrum.
  for (int k = 0; k < kNumMfcc; ++k) {
    float acc = 0.0f;
    const float* d = &dct_[static_cast<std::size_t>(k) * kNumMelBins];
    for (int m = 0; m < kNumMelBins; ++m) acc += d[m] * out.log_mel[m];
    out.mfcc[k] = acc;
  }
}

}  // namespace taver::dsp
