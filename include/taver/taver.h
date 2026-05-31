// Taver public C API — phoneme-level pronunciation assessment.
#ifndef TAVER_TAVER_H_
#define TAVER_TAVER_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(TAVER_BUILD_DLL)
#    define TAVER_API __declspec(dllexport)
#  elif defined(TAVER_USE_DLL)
#    define TAVER_API __declspec(dllimport)
#  else
#    define TAVER_API
#  endif
#else
#  define TAVER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum TaverStatus {
  TAVER_OK = 0,
  TAVER_ERR_INVALID_ARG = 1,
  TAVER_ERR_MODEL_LOAD = 2,
  TAVER_ERR_AUDIO_DEVICE = 3,
  TAVER_ERR_NO_TARGET = 4,
  TAVER_ERR_RUNTIME = 5,
  TAVER_ERR_FILE = 6
} TaverStatus;

typedef enum TaverBackend {
  TAVER_BACKEND_AUTO = 0,
  TAVER_BACKEND_CPU = 1,
  TAVER_BACKEND_CUDA = 2,
  TAVER_BACKEND_DIRECTML = 3
} TaverBackend;

typedef enum TaverErrorType {
  TAVER_PHONE_GOOD = 0,
  TAVER_PHONE_FAIR = 1,
  TAVER_PHONE_SUBSTITUTED = 2,
  TAVER_PHONE_DELETED = 3,
  TAVER_PHONE_INSERTED = 4
} TaverErrorType;

#define TAVER_MAX_PHONEMES 128
#define TAVER_MAX_WORDS    48
#define TAVER_PHONEME_LEN  16
#define TAVER_WORD_LEN     40

typedef struct TaverPhonemeScore {
  char  phoneme[TAVER_PHONEME_LEN];
  char  realized[TAVER_PHONEME_LEN];
  float score;
  int   error_type;
  float start_sec;
  float duration_sec;
} TaverPhonemeScore;

typedef struct TaverWordScore {
  char  text[TAVER_WORD_LEN];
  float score;
  int   phoneme_start;
  int   phoneme_count;
  int   has_error;
} TaverWordScore;

typedef struct TaverResult {
  float overall_score;
  int   num_phonemes;
  TaverPhonemeScore phonemes[TAVER_MAX_PHONEMES];
  int   num_words;
  TaverWordScore words[TAVER_MAX_WORDS];
  float utterance_sec;
  float latency_ms;
  float speaking_rate_pps;
  float mean_pitch_hz;
  float pitch_range_hz;
  float fluency_score;
} TaverResult;

typedef struct TaverMetrics {
  double e2e_latency_ms_p50;
  double e2e_latency_ms_p95;
  double e2e_latency_ms_max;
  double inference_ms_avg;
  uint64_t utterances_scored;
  uint64_t audio_chunks_dropped;
  size_t   peak_working_set_bytes;
  size_t   pool_blocks_in_use;
} TaverMetrics;

typedef void (*TaverResultCallback)(const TaverResult* result, void* user);

typedef struct TaverConfig {
  const char* model_path;
  const char* vocab_path;
  const char* espeak_path;
  const char* lexicon_path;      // optional word->phoneme overrides (data/lexicon.txt)
  const char* calibration_path;  // optional GOP tuning (config/calibration.json)
  int         backend;
  int         g2p_voice;         // 0=en-us, 1=en-gb
  int         print_features;
  TaverResultCallback on_result;
  void*       user_data;
} TaverConfig;

typedef struct TaverEngine TaverEngine;

TAVER_API const char*   taver_version(void);
TAVER_API const char*   taver_status_string(TaverStatus s);
TAVER_API TaverStatus   taver_create(const TaverConfig* cfg, TaverEngine** out);
TAVER_API void          taver_destroy(TaverEngine* eng);

TAVER_API TaverStatus taver_set_target(TaverEngine* eng, const char* phoneme_string);
TAVER_API TaverStatus taver_set_target_text(TaverEngine* eng, const char* text);
TAVER_API void        taver_set_g2p_voice(TaverEngine* eng, int voice);
TAVER_API TaverStatus taver_get_target_info(TaverEngine* eng,
                                            char* phonemes_out, size_t phonemes_cap,
                                            char* ipa_out, size_t ipa_cap);

TAVER_API TaverStatus taver_start(TaverEngine* eng);
TAVER_API void        taver_stop(TaverEngine* eng);

TAVER_API TaverStatus taver_score_wav(TaverEngine* eng, const char* wav_path, TaverResult* out);
TAVER_API TaverStatus taver_score_samples(TaverEngine* eng, const float* samples,
                                          int num_samples, TaverResult* out);

TAVER_API TaverStatus taver_get_metrics(TaverEngine* eng, TaverMetrics* out);

#ifdef __cplusplus
}
#endif

#endif  // TAVER_TAVER_H_
