#pragma once

namespace taver::align {

struct GopCalibration {
  float gop_tau = 1.6f;
  float fair_threshold = 75.0f;
  float substituted_cap = 49.0f;
  float deleted_cap = 15.0f;
  float silence_lp = -3.5f;
};

bool load_calibration(const char* path, GopCalibration& out);

}  // namespace taver::align
