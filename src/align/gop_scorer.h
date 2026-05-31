// Layer D: forced alignment + Goodness-of-Pronunciation scoring.
//
// Given the frame-level phoneme log-probabilities from the acoustic model and
// the expected phoneme sequence, we:
//   1. run a CTC forced alignment (Viterbi over the blank-extended target),
//   2. assign model frames to each target phoneme,
//   3. compute a per-phoneme GOP = mean_t( logP(target|t) - max_q logP(q|t) ),
//   4. classify each phoneme (good / fair / substituted / deleted) and map the
//      GOP to a 0..100 score.
// All trellis buffers are pre-sized to the configured maxima; scoring performs
// no large allocations.
#pragma once

#include <string>
#include <vector>

#include "taver/config.h"
#include "taver/types.h"
#include "align/calibration.h"
#include "align/vocabulary.h"

namespace taver::align {

enum class PhoneError { kGood = 0, kFair = 1, kSubstituted = 2, kDeleted = 3, kInserted = 4 };

struct PhonemeScore {
  int target_id = -1;
  std::string symbol;
  std::string realized;
  float score = 0.0f;       // 0..100
  PhoneError error = PhoneError::kGood;
  int start_frame = 0;
  int end_frame = 0;        // exclusive
  float start_sec = 0.0f;
  float duration_sec = 0.0f;
};

struct AlignResult {
  float overall_score = 0.0f;
  int num_frames = 0;
  std::vector<PhonemeScore> phones;
};

class GopScorer {
 public:
  explicit GopScorer(const Vocabulary& vocab);
  void set_calibration(const GopCalibration& c) { cal_ = c; }

  // `target_ids` is the expected phoneme id sequence. `utterance_samples` is the
  // raw 16 kHz sample count of the scored audio (used to convert frames->seconds).
  bool score(const LogitMatrix& logits, const std::vector<int>& target_ids,
             int utterance_samples, AlignResult& out);

 private:
  // Two phonemes in the same equivalence class (e.g. the vowel-length pair
  // i / iː, or the rhotic pair ɹ / r) are not counted as gross substitutions.
  // The GOP numerator uses the best log-prob across the target's class, so a
  // learner who produces an acceptable variant is not unfairly penalised.
  int class_of(int id) const {
    return (id >= 0 && id < static_cast<int>(class_.size())) ? class_[id] : id;
  }
  void build_equivalences();

  const Vocabulary& vocab_;
  GopCalibration cal_;
  std::vector<int> class_;       // id -> canonical class id
  std::vector<float> trellis_;   // (2L+1) * F
  std::vector<int> backptr_;     // (2L+1) * F
};

}  // namespace taver::align
