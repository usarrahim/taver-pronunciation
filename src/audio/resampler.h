// Streaming sample-rate converter + down-mixer.
//
// Converts arbitrary-rate, multi-channel interleaved float audio to mono 16 kHz.
// Uses a windowed-sinc kernel with the cutoff placed at the lower of the input
// and output Nyquist frequencies, so down-sampling is properly anti-aliased
// (important: feeding aliased audio to wav2vec2 corrupts the phoneme posteriors).
// State (recent input samples + fractional read position) is preserved across
// calls so chunk boundaries are seamless.
#pragma once

#include <vector>

namespace taver::audio {

class Resampler {
 public:
  Resampler(int in_rate, int in_channels, int out_rate);

  // Downmix + resample `frames` interleaved input frames. Resulting mono 16 kHz
  // samples are appended to `out`.
  void process(const float* interleaved, int frames, std::vector<float>& out);

  // Flush remaining tail (call once at end of a finite stream).
  void flush(std::vector<float>& out);

  void reset();

 private:
  void emit_available(std::vector<float>& out);

  int in_rate_;
  int in_channels_;
  int out_rate_;
  double ratio_;        // in_rate / out_rate (input samples per output sample)
  int half_taps_;       // kernel radius in input samples
  double cutoff_;       // normalized cutoff (cycles/input-sample)

  std::vector<float> history_;  // recent mono input samples
  double read_pos_;             // fractional position into history_ origin
  long long produced_;          // diagnostics
};

}  // namespace taver::audio
