// Utterance-level prosody / fluency features.
//
// These are cheap, classic DSP estimates computed once per scored utterance (not
// on the hot path): voiced pitch (autocorrelation F0), speaking rate, and a
// fluency score derived from internal pauses and rate plausibility. They give
// the learner feedback beyond raw phoneme accuracy (monotone, too fast, choppy).
#pragma once

namespace taver::prosody {

struct ProsodyStats {
  float mean_pitch_hz = 0.0f;
  float pitch_range_hz = 0.0f;      // p90 - p10 of voiced F0
  float speaking_rate_pps = 0.0f;   // phonemes per second of voiced speech
  float fluency_score = 0.0f;       // 0..100
};

// `x` is mono 16 kHz float audio of the (already silence-trimmed) utterance.
void analyze(const float* x, int n, int sample_rate, int num_phonemes,
             ProsodyStats& out);

}  // namespace taver::prosody
