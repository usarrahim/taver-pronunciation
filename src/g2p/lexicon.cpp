#include "g2p/lexicon.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace taver::g2p {

bool Lexicon::load(const std::string& path, std::string& err) {
  map_.clear();
  if (path.empty()) return true;
  std::ifstream f(path);
  if (!f) {
    err = "lexicon not found: " + path;
    return false;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string word, ph;
    if (!(ss >> word)) continue;
    std::string rest;
    std::getline(ss, rest);
    while (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
    if (rest.empty()) continue;
    for (char& c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    map_[word] = rest;
  }
  return true;
}

bool Lexicon::lookup(const std::string& word_lower, std::string& phonemes) const {
  const auto it = map_.find(word_lower);
  if (it == map_.end()) return false;
  phonemes = it->second;
  return true;
}

}  // namespace taver::g2p
