// Minimal in-place iterative radix-2 Cooley-Tukey FFT.
//
// We deliberately avoid a third-party FFT dependency: the transform sizes used
// here are small powers of two and a hand-rolled FFT keeps the build trivial
// and the binary tiny (an edge-computing goal). Twiddle factors and the
// bit-reversal permutation are precomputed once; computing a transform performs
// zero allocation.
#pragma once

#include <cstddef>
#include <vector>

namespace taver::dsp {

class Fft {
 public:
  explicit Fft(std::size_t n);

  std::size_t size() const { return n_; }

  // In-place complex FFT. `re`/`im` are length-n arrays. inverse=false => DFT.
  void transform(float* re, float* im, bool inverse) const;

  // Convenience: real input -> power spectrum (|X|^2) of the first n/2+1 bins.
  // `in` has `size()` samples; `power` must hold size()/2 + 1 values.
  // Uses the supplied scratch buffers (no allocation).
  void power_spectrum(const float* in, float* power,
                      float* scratch_re, float* scratch_im) const;

 private:
  std::size_t n_;
  unsigned log2n_;
  std::vector<std::size_t> rev_;  // bit-reversal table
  std::vector<float> cos_;        // twiddle cos, length n/2
  std::vector<float> sin_;        // twiddle sin, length n/2
};

}  // namespace taver::dsp
