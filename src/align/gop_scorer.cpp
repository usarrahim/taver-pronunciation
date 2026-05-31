#include "align/gop_scorer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace taver::align {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// GOP (in nats, <= 0) -> 0..100. Tuned so a confidently-correct phoneme scores
// ~100 and a clearly-wrong one collapses toward 0.
float gop_to_score(float gop, float tau) {
  float s = 100.0f * std::exp(gop / tau);
  if (s > 100.0f) s = 100.0f;
  if (s < 0.0f) s = 0.0f;
  return s;
}
}  // namespace

GopScorer::GopScorer(const Vocabulary& vocab) : vocab_(vocab) {
  const int s_max = 2 * kMaxTargetPhonemes + 1;
  trellis_.assign(static_cast<std::size_t>(s_max) * kMaxModelFrames, kNegInf);
  backptr_.assign(static_cast<std::size_t>(s_max) * kMaxModelFrames, -1);
  build_equivalences();
}

void GopScorer::build_equivalences() {
  const int n = vocab_.size();
  class_.resize(n);
  for (int i = 0; i < n; ++i) class_[i] = i;  // each phoneme is its own class

  // Acceptable variants in the eSpeak IPA inventory: long/short vowel pairs and
  // a few near-identical consonants. Members of a group are treated as mutually
  // acceptable realisations of one another.
  static const std::vector<std::vector<const char*>> groups = {
      {"iː", "i"}, {"uː", "u"}, {"ɔː", "ɔ"}, {"ɑː", "ɑ"}, {"eː", "e"},
      {"oː", "o"}, {"ɜː", "ɜ"}, {"ɚ", "ɝ", "ɜ"}, {"ɹ", "r"}, {"ɡ", "g"},
      {"ɾ", "ɹ"}, {"ʔ", "t"},
  };
  for (const auto& g : groups) {
    int canon = -1;
    for (const char* sym : g) {
      const int id = vocab_.id(sym);
      if (id < 0) continue;
      if (canon < 0) canon = id;
      class_[id] = canon;
    }
  }
}

bool GopScorer::score(const LogitMatrix& logits, const std::vector<int>& target_ids,
                      int utterance_samples, AlignResult& out) {
  const int F = logits.num_frames;
  const int L = static_cast<int>(target_ids.size());
  out.phones.clear();
  out.num_frames = F;
  if (F <= 0 || L <= 0 || L > kMaxTargetPhonemes) return false;

  const int blank = vocab_.blank_id();
  const int S = 2 * L + 1;

  // Extended label for each state: even -> blank, odd -> target phoneme.
  auto ext_label = [&](int s) -> int {
    return (s % 2 == 0) ? blank : target_ids[(s - 1) / 2];
  };

  auto T = [&](int f, int s) -> float& {
    return trellis_[static_cast<std::size_t>(f) * S + s];
  };
  auto B = [&](int f, int s) -> int& {
    return backptr_[static_cast<std::size_t>(f) * S + s];
  };
  auto emit = [&](int f, int s) -> float {
    const int lbl = ext_label(s);
    return (lbl < logits.vocab_size) ? logits.row(f)[lbl] : kNegInf;
  };

  // --- Viterbi forced alignment over the CTC trellis ---
  for (int f = 0; f < F; ++f)
    for (int s = 0; s < S; ++s) { T(f, s) = kNegInf; B(f, s) = -1; }

  T(0, 0) = emit(0, 0);
  if (S > 1) T(0, 1) = emit(0, 1);

  for (int f = 1; f < F; ++f) {
    for (int s = 0; s < S; ++s) {
      float best = T(f - 1, s);
      int arg = s;
      if (s >= 1 && T(f - 1, s - 1) > best) { best = T(f - 1, s - 1); arg = s - 1; }
      // CTC skip: only into a label state whose phoneme differs from two back.
      if (s >= 2 && (s % 2 == 1) && ext_label(s) != ext_label(s - 2) &&
          T(f - 1, s - 2) > best) {
        best = T(f - 1, s - 2);
        arg = s - 2;
      }
      if (best == kNegInf) { T(f, s) = kNegInf; B(f, s) = -1; continue; }
      T(f, s) = best + emit(f, s);
      B(f, s) = arg;
    }
  }

  // Terminate at the last label or the trailing blank, whichever is better.
  int end_state = S - 1;
  if (S >= 2 && T(F - 1, S - 2) > T(F - 1, S - 1)) end_state = S - 2;
  if (T(F - 1, end_state) == kNegInf) {
    // Degenerate alignment (e.g. far too few frames). Fall back to even spread.
    end_state = std::min(S - 1, 2 * L);
  }

  // Backtrack: state per frame.
  std::vector<int> state_at(F, 0);
  int s = end_state;
  for (int f = F - 1; f >= 0; --f) {
    state_at[f] = s;
    if (f > 0) {
      int prev = B(f, s);
      s = (prev >= 0) ? prev : s;
    }
  }

  // --- Per-phoneme GOP from frames whose state is that phoneme's label ---
  const float frame_sec =
      (F > 0) ? (static_cast<float>(utterance_samples) / F) / 16000.0f : 0.02f;

  double overall = 0.0;
  int counted = 0;
  for (int k = 0; k < L; ++k) {
    const int label_state = 2 * k + 1;
    PhonemeScore ps;
    ps.target_id = target_ids[k];
    ps.symbol = vocab_.symbol(target_ids[k]);

    // Members of the target phoneme's equivalence class.
    const int target_class = class_of(target_ids[k]);
    std::vector<int> members;
    for (int v = 0; v < logits.vocab_size; ++v)
      if (class_of(v) == target_class) members.push_back(v);

    // Collect the aligned span, then (because CTC posteriors are peaky and
    // blank-dominated) score from the PEAK evidence inside it rather than a
    // per-frame average:
    //   * best_target_lp  = strongest log-prob of the target's class anywhere,
    //   * best_comp_lp/id = strongest *non-blank* competing sound anywhere.
    // GOP = best_target_lp - best_comp_lp  (<= 0). If the target sound is what
    // fired strongest, GOP ~ 0 -> high score.
    int first = -1, last = -1, frames = 0;
    float best_target_lp = -1e30f;
    float best_comp_lp = -1e30f;
    int best_comp_id = -1;

    for (int f = 0; f < F; ++f) {
      if (state_at[f] != label_state) continue;
      if (first < 0) first = f;
      last = f;
      ++frames;
    }
    if (frames == 0) {  // should not happen given monotonic alignment
      ps.error = PhoneError::kDeleted; ps.score = 0.0f; ps.realized = "-";
      out.phones.push_back(ps);
      continue;
    }
    // Search the span plus one neighbouring frame each side so a spike on the
    // boundary is not missed.
    const int lo = std::max(0, first - 1);
    const int hi = std::min(F - 1, last + 1);
    for (int f = lo; f <= hi; ++f) {
      const float* row = logits.row(f);
      for (int m : members) best_target_lp = std::max(best_target_lp, row[m]);
      for (int v = 0; v < logits.vocab_size; ++v) {
        if (v == blank) continue;
        if (row[v] > best_comp_lp) { best_comp_lp = row[v]; best_comp_id = v; }
      }
    }

    ps.start_frame = first;
    ps.end_frame = last + 1;
    ps.start_sec = first * frame_sec;
    ps.duration_sec = (last - first + 1) * frame_sec;

    const float gop = best_target_lp - best_comp_lp;  // <= 0
    ps.score = gop_to_score(gop, cal_.gop_tau);

    const int realized_id = best_comp_id;
    if (best_comp_lp < cal_.silence_lp) {
      ps.error = PhoneError::kDeleted;
      ps.realized = "-";
      ps.score = std::min(ps.score, cal_.deleted_cap);
    } else if (class_of(realized_id) != target_class) {
      ps.error = PhoneError::kSubstituted;
      ps.realized = vocab_.symbol(realized_id);
      if (ps.score > cal_.substituted_cap) ps.score = cal_.substituted_cap;
    } else {
      ps.realized = vocab_.symbol(realized_id);
      ps.error = (ps.score >= cal_.fair_threshold) ? PhoneError::kGood : PhoneError::kFair;
    }

    overall += ps.score;
    ++counted;
    out.phones.push_back(ps);
  }

  out.overall_score = counted ? static_cast<float>(overall / counted) : 0.0f;
  return true;
}

}  // namespace taver::align
