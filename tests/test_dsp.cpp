// Numerical correctness tests for the FFT and feature extractor.
#include <cmath>
#include <cstdio>
#include <vector>

#include "dsp/fft.h"
#include "dsp/feature_extractor.h"

static int failures = 0;
#define CHECK(cond, msg)                                  \
  do {                                                    \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } \
  } while (0)

// A pure tone must produce a single dominant power-spectrum bin at its frequency.
static void test_fft_tone() {
  constexpr int N = 512;
  constexpr int sr = 16000;
  const double freq = 1000.0;
  taver::dsp::Fft fft(N);
  std::vector<float> in(N), re(N), im(N), power(N / 2 + 1);
  for (int i = 0; i < N; ++i)
    in[i] = std::sin(2.0 * M_PI * freq * i / sr);
  fft.power_spectrum(in.data(), power.data(), re.data(), im.data());

  int peak = 0;
  float maxp = 0.0f;
  for (int k = 0; k < N / 2 + 1; ++k)
    if (power[k] > maxp) { maxp = power[k]; peak = k; }
  const double bin_hz = static_cast<double>(sr) / N;
  const double peak_hz = peak * bin_hz;
  CHECK(std::abs(peak_hz - freq) <= bin_hz, "FFT peak at tone frequency");
}

// Round-trip FFT/IFFT should reconstruct the input.
static void test_fft_roundtrip() {
  constexpr int N = 256;
  taver::dsp::Fft fft(N);
  std::vector<float> re(N), im(N, 0.0f), re0(N);
  for (int i = 0; i < N; ++i) re[i] = re0[i] = std::sin(0.3f * i) + 0.5f * std::cos(0.07f * i);
  fft.transform(re.data(), im.data(), false);
  fft.transform(re.data(), im.data(), true);
  double err = 0.0;
  for (int i = 0; i < N; ++i) err += std::abs(re[i] - re0[i]);
  CHECK(err / N < 1e-4, "FFT/IFFT round-trip reconstructs signal");
}

// Higher-pitched tone should shift mel/MFCC energy; just verify finite output.
static void test_features_finite() {
  taver::dsp::FeatureExtractor fe;
  std::vector<float> frame(taver::kFrameLengthSamples);
  for (int i = 0; i < taver::kFrameLengthSamples; ++i)
    frame[i] = 0.2f * std::sin(2.0 * M_PI * 440.0 * i / 16000.0);
  taver::FeatureFrame ff;
  fe.compute(frame.data(), ff);
  bool finite = std::isfinite(ff.energy);
  for (int m = 0; m < taver::kNumMelBins; ++m) finite &= std::isfinite(ff.log_mel[m]);
  for (int c = 0; c < taver::kNumMfcc; ++c) finite &= std::isfinite(ff.mfcc[c]);
  CHECK(finite, "features are finite");
  CHECK(ff.energy > 0.0f, "non-silent frame has positive energy");
}

int main() {
  test_fft_tone();
  test_fft_roundtrip();
  test_features_finite();
  if (failures == 0) std::printf("test_dsp: ALL PASS\n");
  return failures == 0 ? 0 : 1;
}
