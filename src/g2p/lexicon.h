#pragma once

#include <string>
#include <unordered_map>

namespace taver::g2p {

// Optional word -> space-separated phoneme symbols (model vocabulary).
class Lexicon {
 public:
  bool load(const std::string& path, std::string& err);
  bool lookup(const std::string& word_lower, std::string& phonemes) const;
  bool empty() const { return map_.empty(); }

 private:
  std::unordered_map<std::string, std::string> map_;
};

}  // namespace taver::g2p
