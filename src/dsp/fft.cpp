#include "dsp/fft.h"

#include <cassert>
#include <cmath>

namespace taver::dsp {

namespace {
unsigned ilog2(std::size_t n) {
  unsigned l = 0;
  while ((std::size_t{1} << l) < n) ++l;
  return l;
}
}  // namespace

Fft::Fft(std::size_t n) : n_(n), log2n_(ilog2(n)) {
  assert((n_ & (n_ - 1)) == 0 && "FFT size must be a power of two");

  rev_.resize(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    std::size_t r = 0;
    for (unsigned b = 0; b < log2n_; ++b) {
      if (i & (std::size_t{1} << b)) r |= (std::size_t{1} << (log2n_ - 1 - b));
    }
    rev_[i] = r;
  }

  cos_.resize(n_ / 2);
  sin_.resize(n_ / 2);
  for (std::size_t k = 0; k < n_ / 2; ++k) {
    const double ang = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n_);
    cos_[k] = static_cast<float>(std::cos(ang));
    sin_[k] = static_cast<float>(std::sin(ang));
  }
}

void Fft::transform(float* re, float* im, bool inverse) const {
  // Bit-reversal reordering.
  for (std::size_t i = 0; i < n_; ++i) {
    const std::size_t j = rev_[i];
    if (j > i) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }

  const float sign = inverse ? -1.0f : 1.0f;  // conjugate twiddles for IFFT
  for (std::size_t len = 2; len <= n_; len <<= 1) {
    const std::size_t half = len >> 1;
    const std::size_t step = n_ / len;  // index into twiddle tables
    for (std::size_t i = 0; i < n_; i += len) {
      std::size_t tw = 0;
      for (std::size_t j = 0; j < half; ++j, tw += step) {
        const float wr = cos_[tw];
        const float wi = sign * sin_[tw];
        const std::size_t a = i + j;
        const std::size_t b = a + half;
        const float tr = wr * re[b] - wi * im[b];
        const float ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }

  if (inverse) {
    const float inv = 1.0f / static_cast<float>(n_);
    for (std::size_t i = 0; i < n_; ++i) {
      re[i] *= inv;
      im[i] *= inv;
    }
  }
}

void Fft::power_spectrum(const float* in, float* power, float* scratch_re,
                         float* scratch_im) const {
  for (std::size_t i = 0; i < n_; ++i) {
    scratch_re[i] = in[i];
    scratch_im[i] = 0.0f;
  }
  transform(scratch_re, scratch_im, /*inverse=*/false);
  const std::size_t bins = n_ / 2 + 1;
  for (std::size_t k = 0; k < bins; ++k) {
    power[k] = scratch_re[k] * scratch_re[k] + scratch_im[k] * scratch_im[k];
  }
}

}  // namespace taver::dsp
