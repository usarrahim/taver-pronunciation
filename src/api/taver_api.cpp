#include "taver/taver.h"

#include <cstdio>
#include <new>
#include <string>

#include "pipeline/engine.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <psapi.h>
#  pragma comment(lib, "psapi.lib")
#endif

using taver::Engine;

struct TaverEngine {
  Engine impl;
};

namespace {
size_t peak_working_set() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return pmc.PeakWorkingSetSize;
#endif
  return 0;
}
}  // namespace

extern "C" {

TAVER_API const char* taver_version(void) { return "Taver 1.0.0"; }

TAVER_API const char* taver_status_string(TaverStatus s) {
  switch (s) {
    case TAVER_OK: return "ok";
    case TAVER_ERR_INVALID_ARG: return "invalid argument";
    case TAVER_ERR_MODEL_LOAD: return "model load failed";
    case TAVER_ERR_AUDIO_DEVICE: return "audio device error";
    case TAVER_ERR_NO_TARGET: return "no target phoneme sequence set";
    case TAVER_ERR_RUNTIME: return "runtime error";
    case TAVER_ERR_FILE: return "file error";
    default: return "unknown";
  }
}

TAVER_API TaverStatus taver_create(const TaverConfig* cfg, TaverEngine** out) {
  if (!cfg || !out) return TAVER_ERR_INVALID_ARG;
  auto* e = new (std::nothrow) TaverEngine();
  if (!e) return TAVER_ERR_RUNTIME;
  const TaverStatus st = e->impl.init(*cfg);
  if (st != TAVER_OK) {
    delete e;
    return st;
  }
  *out = e;
  return TAVER_OK;
}

TAVER_API void taver_destroy(TaverEngine* eng) {
  if (!eng) return;
  eng->impl.stop();
  delete eng;
}

TAVER_API TaverStatus taver_set_target(TaverEngine* eng, const char* phoneme_string) {
  if (!eng || !phoneme_string) return TAVER_ERR_INVALID_ARG;
  return eng->impl.set_target(phoneme_string);
}

TAVER_API TaverStatus taver_set_target_text(TaverEngine* eng, const char* text) {
  if (!eng || !text) return TAVER_ERR_INVALID_ARG;
  return eng->impl.set_target_text(text);
}

TAVER_API void taver_set_g2p_voice(TaverEngine* eng, int voice) {
  if (eng) eng->impl.set_g2p_voice(voice);
}

TAVER_API TaverStatus taver_get_target_info(TaverEngine* eng, char* phonemes_out,
                                            size_t phonemes_cap, char* ipa_out,
                                            size_t ipa_cap) {
  if (!eng) return TAVER_ERR_INVALID_ARG;
  std::string ph, ipa;
  if (!eng->impl.target_info(ph, ipa)) return TAVER_ERR_NO_TARGET;
  if (phonemes_out && phonemes_cap) std::snprintf(phonemes_out, phonemes_cap, "%s", ph.c_str());
  if (ipa_out && ipa_cap) std::snprintf(ipa_out, ipa_cap, "%s", ipa.c_str());
  return TAVER_OK;
}

TAVER_API TaverStatus taver_start(TaverEngine* eng) {
  if (!eng) return TAVER_ERR_INVALID_ARG;
  return eng->impl.start_microphone();
}

TAVER_API void taver_stop(TaverEngine* eng) {
  if (eng) eng->impl.stop();
}

TAVER_API TaverStatus taver_score_wav(TaverEngine* eng, const char* wav_path,
                                      TaverResult* out) {
  if (!eng || !wav_path || !out) return TAVER_ERR_INVALID_ARG;
  return eng->impl.score_wav(wav_path, out);
}

TAVER_API TaverStatus taver_score_samples(TaverEngine* eng, const float* samples,
                                          int num_samples, TaverResult* out) {
  if (!eng || !samples || !out || num_samples <= 0) return TAVER_ERR_INVALID_ARG;
  return eng->impl.score_samples(samples, num_samples, out);
}

TAVER_API TaverStatus taver_get_metrics(TaverEngine* eng, TaverMetrics* out) {
  if (!eng || !out) return TAVER_ERR_INVALID_ARG;
  eng->impl.get_metrics(out);
  out->peak_working_set_bytes = peak_working_set();
  return TAVER_OK;
}

}  // extern "C"
