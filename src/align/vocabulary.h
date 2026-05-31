// Phoneme vocabulary loaded from the exporter's phoneme_vocab.txt.
// Maps between integer class ids (the model's output dimension) and phoneme
// symbols, and records the CTC blank id and the word-delimiter symbol.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace taver::align {

class Vocabulary {
 public:
  bool load(const std::string& path, std::string& err);

  int size() const { return static_cast<int>(symbols_.size()); }
  int blank_id() const { return blank_id_; }

  const std::string& symbol(int id) const {
    static const std::string kUnk = "<unk>";
    return (id >= 0 && id < static_cast<int>(symbols_.size())) ? symbols_[id] : kUnk;
  }

  // Returns id for a symbol, or -1 if unknown.
  int id(const std::string& symbol) const {
    auto it = index_.find(symbol);
    return it == index_.end() ? -1 : it->second;
  }

  bool is_word_delimiter(int id) const { return id == word_delim_id_; }

 private:
  std::vector<std::string> symbols_;
  std::unordered_map<std::string, int> index_;
  int blank_id_ = 0;
  int word_delim_id_ = -1;
};

}  // namespace taver::align
