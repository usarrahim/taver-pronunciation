#include "audio/resampler.h"

#include <algorithm>
#include <cmath>

namespace taver::audio {

namespace {
constexpr double kPi = 3.14159265358979323846;

inline double sinc(double x) {
  if (std::abs(x) < 1e-9) return 1.0;
  const double px = kPi * x;
  return std::sin(px) / px;
}
// Blackman window over normalized position u in [-1, 1].
inline double blackman(double u) {
  const double a = kPi * (u + 1.0);  // map [-1,1] -> [0, 2pi]
  return 0.42 - 0.5 * std::cos(a) + 0.08 * std::cos(2.0 * a);
}
}  // namespace

Resampler::Resampler(int in_rate, int in_channels, int out_rate)
    : in_rate_(in_rate),
      in_channels_(std::max(1, in_channels)),
      out_rate_(out_rate),
      ratio_(static_cast<double>(in_rate) / out_rate),
      read_pos_(0.0),
      produced_(0) {
  half_taps_ = 16;
  cutoff_ = 0.5 * std::min(1.0, static_cast<double>(out_rate) / in_rate);
  // Pre-pad the history with zeros so the first output is centered correctly.
  history_.assign(static_cast<std::size_t>(half_taps_) + 1, 0.0f);
  read_pos_ = half_taps_;
}

void Resampler::reset() {
  history_.assign(static_cast<std::size_t>(half_taps_) + 1, 0.0f);
  read_pos_ = half_taps_;
  produced_ = 0;
}

void Resampler::process(const float* interleaved, int frames,
                        std::vector<float>& out) {
  history_.reserve(history_.size() + frames);
  const float inv_ch = 1.0f / in_channels_;
  for (int f = 0; f < frames; ++f) {
    float acc = 0.0f;
    const float* base = interleaved + static_cast<std::size_t>(f) * in_channels_;
    for (int c = 0; c < in_channels_; ++c) acc += base[c];
    history_.push_back(acc * inv_ch);
  }
  emit_available(out);
}

void Resampler::emit_available(std::vector<float>& out) {
  const int n = static_cast<int>(history_.size());
  // Produce output while a full kernel fits inside the available history.
  while (true) {
    const int center = static_cast<int>(std::floor(read_pos_));
    const int hi = center + half_taps_;
    if (hi >= n) break;  // need more future samples
    const int lo = center - half_taps_ + 1;
    if (lo < 0) {        // should not happen given pre-pad, guard anyway
      read_pos_ += ratio_;
      continue;
    }
    double acc = 0.0;
    double wsum = 0.0;
    for (int k = lo; k <= hi; ++k) {
      const double dx = read_pos_ - k;             // distance in input samples
      const double u = dx / half_taps_;            // window position [-1,1]
      const double w = 2.0 * cutoff_ * sinc(2.0 * cutoff_ * dx) * blackman(u);
      acc += history_[k] * w;
      wsum += w;
    }
    out.push_back(static_cast<float>(wsum > 1e-9 ? acc / wsum : acc));
    read_pos_ += ratio_;
    ++produced_;
  }

  // Trim consumed history to keep memory flat.
  const int keep_from = static_cast<int>(std::floor(read_pos_)) - half_taps_ - 1;
  if (keep_from > 4096) {
    history_.erase(history_.begin(), history_.begin() + keep_from);
    read_pos_ -= keep_from;
  }
}

void Resampler::flush(std::vector<float>& out) {
  // Append a kernel's worth of zeros so trailing samples can be emitted.
  history_.insert(history_.end(), static_cast<std::size_t>(half_taps_ + 1), 0.0f);
  emit_available(out);
}

}  // namespace taver::audio
