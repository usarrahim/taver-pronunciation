// Taver — real-time, on-device pronunciation assessment engine.
// Central compile-time configuration. These values define the contract that
// every layer (audio, DSP, inference, alignment) agrees on. They are kept in
// one place so the pipeline stays internally consistent.
#pragma once

#include <cstddef>
#include <cstdint>

namespace taver {

// ---- Audio / framing -------------------------------------------------------
// The neural acoustic model (wav2vec2-phoneme family) expects 16 kHz mono PCM.
inline constexpr int   kTargetSampleRate   = 16000;
inline constexpr int   kFrameLengthSamples = 400;   // 25 ms analysis window
inline constexpr int   kFrameShiftSamples  = 160;   // 10 ms hop
inline constexpr int   kFftSize            = 512;   // next pow2 >= frame length

// ---- DSP / features --------------------------------------------------------
inline constexpr int   kNumMelBins  = 80;           // log-mel filterbank size
inline constexpr int   kNumMfcc     = 13;           // MFCC coefficients (incl. c0)
inline constexpr float kMelLowHz    = 20.0f;
inline constexpr float kMelHighHz   = 8000.0f;      // Nyquist at 16 kHz
inline constexpr float kPreemphasis = 0.97f;

// ---- Capture buffering -----------------------------------------------------
// Period of the audio callback chunk we copy out of WASAPI (in samples at the
// target rate). 10 ms keeps capture-side latency tiny.
inline constexpr int   kCaptureChunkSamples = 160;  // 10 ms @ 16 kHz

// Maximum length of a single scored utterance (samples). Pre-allocated so the
// inference path never calls malloc. 12 s is comfortably longer than any single
// word or short sentence a learner reads aloud.
inline constexpr int   kMaxUtteranceSamples = kTargetSampleRate * 12;

// Maximum number of acoustic-model output frames for one utterance. wav2vec2
// downsamples the 16 kHz input by ~320x (one frame per 20 ms).
inline constexpr int   kMaxModelFrames = (kMaxUtteranceSamples / 320) + 8;

// Maximum size of the phoneme vocabulary the model can emit.
inline constexpr int   kMaxVocab = 512;

// Maximum number of phonemes in a single target sequence.
inline constexpr int   kMaxTargetPhonemes = 128;

// ---- Voice activity detection ---------------------------------------------
// Energy-based VAD thresholds, tuned for close-talk microphones. Hangover keeps
// short pauses inside a word from prematurely closing the utterance.
inline constexpr float kVadOnsetEnergy   = 4.0e-4f; // mean square to open
inline constexpr float kVadOffsetEnergy  = 1.5e-4f; // mean square to close
inline constexpr int   kVadHangoverFrames = 22;     // ~220 ms of trailing silence
inline constexpr int   kVadMinSpeechFrames = 6;     // ignore < 60 ms blips
inline constexpr int   kKeepTrailingFrames = 8;     // silence kept after offset (~80 ms)

}  // namespace taver
