#include "align/vocabulary.h"

#include <fstream>
#include <sstream>

namespace taver::align {

bool Vocabulary::load(const std::string& path, std::string& err) {
  std::ifstream f(path);
  if (!f) {
    err = "cannot open vocab file: " + path;
    return false;
  }
  symbols_.clear();
  index_.clear();
  std::string word_delim = "|";

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      // Header: "# vocab_size=.. blank_id=N word_delimiter=X"
      std::istringstream hs(line.substr(1));
      std::string tok;
      while (hs >> tok) {
        const auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = tok.substr(0, eq);
        const std::string v = tok.substr(eq + 1);
        if (k == "blank_id") blank_id_ = std::stoi(v);
        else if (k == "word_delimiter") word_delim = v;
      }
      continue;
    }
    // "<id>\t<symbol>"
    const auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    const int id = std::stoi(line.substr(0, tab));
    std::string sym = line.substr(tab + 1);
    // Strip trailing CR (Windows line endings on a *nix-written file).
    while (!sym.empty() && (sym.back() == '\r' || sym.back() == '\n')) sym.pop_back();
    if (static_cast<int>(symbols_.size()) <= id) symbols_.resize(id + 1);
    symbols_[id] = sym;
    index_[sym] = id;
  }

  if (symbols_.empty()) {
    err = "vocab file had no entries";
    return false;
  }
  auto it = index_.find(word_delim);
  word_delim_id_ = (it == index_.end()) ? -1 : it->second;
  return true;
}

}  // namespace taver::align
