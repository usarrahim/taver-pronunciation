#include "g2p/g2p.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace taver::g2p {

namespace {

std::wstring widen(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  if (!w.empty() && w.back() == L'\0') w.pop_back();
  return w;
}

// Split UTF-8 string into a vector of single-code-point UTF-8 substrings.
std::vector<std::string> to_codepoints(const std::string& s) {
  std::vector<std::string> cps;
  std::size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    std::size_t len = 1;
    if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
    if (i + len > s.size()) len = 1;
    cps.push_back(s.substr(i, len));
    i += len;
  }
  return cps;
}

// IPA modifiers we strip before matching (stress, syllable & length helpers that
// are not part of the model's symbol tokens).
bool is_modifier(const std::string& cp) {
  static const std::unordered_set<std::string> mods = {
      "\u02c8",  // ˈ primary stress
      "\u02cc",  // ˌ secondary stress
      "\u02d1",  // ˑ half-long
      "\u0361",  // ͡ tie bar
      "\u200d",  // ZWJ
      ".", " ", "\t",
  };
  return mods.count(cp) > 0;
}

}  // namespace

bool G2P::init(const std::string& espeak_exe, const align::Vocabulary& vocab, std::string& err) {
  vocab_ = &vocab;
  espeak_exe_ = espeak_exe;

  std::ifstream probe(espeak_exe_, std::ios::binary);
  if (!probe) { err = "espeak-ng.exe not found at: " + espeak_exe_; available_ = false; return false; }

  // Data directory is the folder containing the exe.
  const std::size_t slash = espeak_exe_.find_last_of("\\/");
  data_dir_ = (slash == std::string::npos) ? "." : espeak_exe_.substr(0, slash);

  // Build the set of matchable symbols (skip special tokens).
  for (int id = 0; id < vocab.size(); ++id) {
    const std::string& s = vocab.symbol(id);
    if (s.empty() || s[0] == '<') continue;
    symbols_.insert(s);
    const int cps = static_cast<int>(to_codepoints(s).size());
    if (cps > max_sym_cp_) max_sym_cp_ = cps;
  }
  available_ = true;
  return true;
}

bool G2P::run_espeak(const std::string& text, std::string& ipa_out, std::string& err) {
  // Write text to a temp file so no user content ever touches a command line.
  char tmpdir[MAX_PATH];
  GetTempPathA(MAX_PATH, tmpdir);
  char tmpfile[MAX_PATH];
  GetTempFileNameA(tmpdir, "p2p", 0, tmpfile);
  {
    std::ofstream f(tmpfile, std::ios::binary);
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
  }

  // espeak finds its data via ESPEAK_DATA_PATH (inherited by the child).
  _putenv_s("ESPEAK_DATA_PATH", data_dir_.c_str());

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE rd = nullptr, wr = nullptr;
  if (!CreatePipe(&rd, &wr, &sa, 0)) { err = "CreatePipe failed"; DeleteFileA(tmpfile); return false; }
  SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);  // parent read end not inherited

  std::wstring cmd = L"\"" + widen(espeak_exe_) + L"\" -q --ipa -v " + widen(voice_) +
                     L" -f \"" + widen(tmpfile) + L"\"";
  std::vector<wchar_t> cmdbuf(cmd.begin(), cmd.end());
  cmdbuf.push_back(L'\0');

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = wr;
  si.hStdError = wr;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION pi{};

  const BOOL ok = CreateProcessW(nullptr, cmdbuf.data(), nullptr, nullptr, TRUE,
                                 CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  CloseHandle(wr);  // parent must close its copy so reads see EOF
  if (!ok) { CloseHandle(rd); err = "failed to launch espeak-ng"; DeleteFileA(tmpfile); return false; }

  std::string out;
  char buf[4096];
  DWORD n = 0;
  while (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n > 0) out.append(buf, n);
  CloseHandle(rd);

  // Kill hung children; otherwise repeated G2P calls leak processes and crash
  // the host after hundreds of utterances (eval harness / long UI sessions).
  if (WaitForSingleObject(pi.hProcess, 5000) == WAIT_TIMEOUT) {
    TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 2000);
  }
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  DeleteFileA(tmpfile);

  ipa_out = out;
  return true;
}

void G2P::tokenize_word(const std::string& ipa_word, G2PResult& out) {
  // Drop stress/length modifiers, keep the rest as code points.
  std::vector<std::string> cps;
  for (auto& cp : to_codepoints(ipa_word))
    if (!is_modifier(cp)) cps.push_back(cp);

  std::size_t i = 0;
  while (i < cps.size()) {
    bool matched = false;
    const int maxlen = static_cast<int>(std::min<std::size_t>(max_sym_cp_, cps.size() - i));
    for (int len = maxlen; len >= 1 && !matched; --len) {
      std::string cand;
      for (int k = 0; k < len; ++k) cand += cps[i + k];
      if (symbols_.count(cand)) {
        out.phoneme_ids.push_back(vocab_->id(cand));
        i += len;
        matched = true;
      }
    }
    if (!matched) { ++out.dropped; ++i; }  // unknown diacritic etc.
  }
}

void G2P::set_voice(int g2p_voice) { voice_ = (g2p_voice == 1) ? "en-gb" : "en-us"; }

bool G2P::convert(const std::string& text, G2PResult& out, std::string& err) {
  out = G2PResult{};
  if (!available_) { err = "G2P unavailable (espeak not initialised)"; return false; }

  // Single-word lexicon override (demo words, curated pronunciations).
  if (lexicon_ && !lexicon_->empty()) {
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    std::string ph;
    if (lexicon_->lookup(lower, ph)) {
      out.words.push_back(lower);
      out.word_start.push_back(0);
      std::istringstream ps(ph);
      std::string tok;
      while (ps >> tok) {
        const int id = vocab_->id(tok);
        if (id >= 0) out.phoneme_ids.push_back(id);
        else ++out.dropped;
      }
      if (!out.phoneme_ids.empty()) {
        out.ipa = ph;
        return true;
      }
    }
  }

  // Lowercase for phonemisation: espeak treats short all-caps tokens as
  // initialisms ("IT" -> "eye-tee"), which mangles upper-cased prompts. The
  // original text is still used for the orthographic word labels below.
  std::string lower = text;
  for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  std::string ipa;
  if (!run_espeak(lower, ipa, err)) return false;
  // Collapse newlines (clause breaks) to spaces.
  for (char& c : ipa) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
  out.ipa = ipa;

  // Orthographic words (for labels) from the input text.
  {
    std::istringstream ws(text);
    std::string w;
    while (ws >> w) {
      std::string clean;
      for (char c : w) if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') clean += c;
      if (!clean.empty()) out.words.push_back(clean);
    }
  }

  // Phonemes per IPA word; record where each word starts. espeak emits one
  // space-separated token per *spoken* word, which can differ from the count of
  // orthographic words (it merges function words, e.g. "was a" -> "wʌzɐ").
  std::istringstream is(ipa);
  std::string ipa_word;
  while (is >> ipa_word) {
    out.word_start.push_back(static_cast<int>(out.phoneme_ids.size()));
    tokenize_word(ipa_word, out);
  }

  if (out.phoneme_ids.empty()) { err = "no phonemes produced for: " + text; return false; }

  // Reconcile orthographic labels with the spoken word groups so the two arrays
  // are always the same length (callers index them in lockstep).
  const int n_groups = static_cast<int>(out.word_start.size());
  const int n_ortho = static_cast<int>(out.words.size());
  if (n_ortho != n_groups) {
    std::vector<std::string> lbl(n_groups);
    for (int i = 0; i < n_groups && i < n_ortho; ++i) lbl[i] = out.words[i];
    // If espeak merged trailing words, fold the leftovers into the last label.
    if (n_ortho > n_groups && n_groups > 0)
      for (int j = n_groups; j < n_ortho; ++j) lbl[n_groups - 1] += " " + out.words[j];
    out.words.swap(lbl);
  }
  return true;
}

}  // namespace taver::g2p
