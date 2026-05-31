#include "prosody/prosody.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace taver::prosody {

namespace {
float percentile(std::vector<float>& v, float p) {
  if (v.empty()) return 0.0f;
  std::sort(v.begin(), v.end());
  const std::size_t idx = std::min(v.size() - 1,
      static_cast<std::size_t>(p * (v.size() - 1)));
  return v[idx];
}
}  // namespace

void analyze(const float* x, int n, int sr, int num_phonemes, ProsodyStats& out) {
  out = ProsodyStats{};
  if (!x || n < sr / 20) return;  // < 50 ms: nothing meaningful

  const int win = sr * 40 / 1000;   // 40 ms analysis window
  const int hop = sr * 20 / 1000;   // 20 ms hop
  const int min_lag = sr / 400;     // 400 Hz ceiling
  const int max_lag = sr / 70;      // 70 Hz floor

  // Global energy reference to separate voiced frames from internal pauses.
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) sum_sq += static_cast<double>(x[i]) * x[i];
  const float rms = static_cast<float>(std::sqrt(sum_sq / n));
  const float voiced_thr = std::max(1e-4f, rms * 0.5f);

  std::vector<float> f0s;
  int total_frames = 0, voiced_frames = 0, pause_frames = 0;
  int longest_pause = 0, cur_pause = 0;

  for (int start = 0; start + win <= n; start += hop) {
    ++total_frames;
    const float* w = x + start;

    double fe = 0.0;
    for (int i = 0; i < win; ++i) fe += static_cast<double>(w[i]) * w[i];
    const float frame_rms = static_cast<float>(std::sqrt(fe / win));

    if (frame_rms < voiced_thr) {  // silence / unvoiced gap
      ++pause_frames;
      cur_pause += 1;
      longest_pause = std::max(longest_pause, cur_pause);
      continue;
    }
    cur_pause = 0;

    // Autocorrelation pitch estimate.
    double r0 = 0.0;
    for (int i = 0; i < win; ++i) r0 += static_cast<double>(w[i]) * w[i];
    if (r0 < 1e-9) continue;
    double best = 0.0;
    int best_lag = 0;
    for (int lag = min_lag; lag <= max_lag && lag < win; ++lag) {
      double r = 0.0;
      for (int i = 0; i + lag < win; ++i) r += static_cast<double>(w[i]) * w[i + lag];
      if (r > best) { best = r; best_lag = lag; }
    }
    if (best_lag > 0 && best / r0 > 0.35) {  // clear periodicity => voiced
      ++voiced_frames;
      f0s.push_back(static_cast<float>(sr) / best_lag);
    }
  }

  if (!f0s.empty()) {
    double s = 0.0;
    for (float f : f0s) s += f;
    out.mean_pitch_hz = static_cast<float>(s / f0s.size());
    std::vector<float> tmp = f0s;
    out.pitch_range_hz = percentile(tmp, 0.9f) - percentile(f0s, 0.1f);
  }

  const float voiced_sec = std::max(0.1f, voiced_frames * hop / static_cast<float>(sr));
  if (num_phonemes > 0)
    out.speaking_rate_pps = num_phonemes / voiced_sec;

  // Fluency: start at 100, penalise long internal pauses and implausible rate.
  float fluency = 100.0f;
  const float pause_ratio =
      total_frames > 0 ? pause_frames / static_cast<float>(total_frames) : 0.0f;
  fluency -= std::min(40.0f, pause_ratio * 80.0f);              // choppy / hesitant
  const float longest_pause_sec = longest_pause * hop / static_cast<float>(sr);
  if (longest_pause_sec > 0.4f) fluency -= std::min(25.0f, (longest_pause_sec - 0.4f) * 40.0f);
  const float rate = out.speaking_rate_pps;
  if (rate > 0.0f && (rate < 5.0f || rate > 17.0f)) {
    const float d = (rate < 5.0f) ? (5.0f - rate) : (rate - 17.0f);
    fluency -= std::min(20.0f, d * 5.0f);
  }
  out.fluency_score = std::clamp(fluency, 0.0f, 100.0f);
}

}  // namespace taver::prosody
