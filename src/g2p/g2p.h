// Grapheme-to-phoneme: turn arbitrary English text into the model's phoneme ids.
//
// We shell out to the bundled espeak-ng (the same engine whose phoneme set the
// acoustic model was trained on), get IPA, then greedily tokenise it into the
// model vocabulary's symbols. G2P runs once per target (never on the audio hot
// path), so a short subprocess call is perfectly fine.
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "align/vocabulary.h"
#include "g2p/lexicon.h"

namespace taver::g2p {

struct G2PResult {
  std::vector<int> phoneme_ids;          // model class ids, in order
  std::vector<int> word_start;           // phoneme index where each word begins
  std::vector<std::string> words;        // orthographic words (for the report)
  std::string ipa;                       // raw espeak IPA (debug/display)
  int dropped = 0;                       // IPA symbols not in the vocabulary
};

class G2P {
 public:
  // `espeak_exe` is the full path to espeak-ng.exe; the data directory is taken
  // from the same folder. Returns false (with `err`) if espeak is unavailable.
  bool init(const std::string& espeak_exe, const align::Vocabulary& vocab, std::string& err);
  void set_voice(int g2p_voice);  // 0=en-us, 1=en-gb
  void set_lexicon(Lexicon* lex) { lexicon_ = lex; }

  bool available() const { return available_; }

  bool convert(const std::string& text, G2PResult& out, std::string& err);

 private:
  std::string espeak_exe_;
  std::string data_dir_;
  const align::Vocabulary* vocab_ = nullptr;
  std::unordered_set<std::string> symbols_;  // valid vocab symbols
  int max_sym_cp_ = 1;
  bool available_ = false;
  std::string voice_ = "en-us";
  Lexicon* lexicon_ = nullptr;

  bool run_espeak(const std::string& text, std::string& ipa_out, std::string& err);
  void tokenize_word(const std::string& ipa_word, G2PResult& out);
};

}  // namespace taver::g2p
