#include "align/calibration.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace taver::align {

namespace {
float read_num(const std::string& key, const std::string& json, float def) {
  const auto pos = json.find('"' + key + '"');
  if (pos == std::string::npos) return def;
  auto p = json.find(':', pos);
  if (p == std::string::npos) return def;
  return static_cast<float>(std::atof(json.c_str() + p + 1));
}
}  // namespace

bool load_calibration(const char* path, GopCalibration& out) {
  if (!path || !path[0]) return true;
  std::ifstream f(path);
  if (!f) return false;
  std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  out.gop_tau = read_num("gop_tau", json, out.gop_tau);
  out.fair_threshold = read_num("fair_threshold", json, out.fair_threshold);
  out.substituted_cap = read_num("substituted_cap", json, out.substituted_cap);
  out.deleted_cap = read_num("deleted_cap", json, out.deleted_cap);
  out.silence_lp = read_num("silence_lp", json, out.silence_lp);
  return true;
}

}  // namespace taver::align
