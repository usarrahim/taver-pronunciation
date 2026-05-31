// Integration test for grapheme-to-phoneme: text -> vocab phonemes + word bounds.
//
// Requires the model vocabulary and the bundled espeak-ng to be present (they
// are after scripts\setup.ps1). If either is missing (e.g. a fresh checkout in
// CI), the test SKIPS rather than fails, so it never blocks an unrelated build.
#include <cstdio>
#include <string>

#include "align/vocabulary.h"
#include "g2p/g2p.h"

static int failures = 0;
#define CHECK(cond, msg)                                            \
  do {                                                              \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; }    \
  } while (0)

static std::string join(const taver::align::Vocabulary& v,
                        const std::vector<int>& ids) {
  std::string s;
  for (int id : ids) { if (!s.empty()) s += ' '; s += v.symbol(id); }
  return s;
}

int main() {
  taver::align::Vocabulary vocab;
  std::string err;
  if (!vocab.load("models/phoneme_vocab.txt", err)) {
    std::printf("SKIP test_g2p: vocab not found (%s)\n", err.c_str());
    return 0;
  }

  taver::g2p::G2P g2p;
  if (!g2p.init("third_party/espeak/eSpeak NG/espeak-ng.exe", vocab, err) ||
      !g2p.available()) {
    std::printf("SKIP test_g2p: espeak unavailable (%s)\n", err.c_str());
    return 0;
  }

  taver::g2p::G2PResult r;
  CHECK(g2p.convert("three", r, err), "convert 'three'");
  CHECK(join(vocab, r.phoneme_ids) == "\xCE\xB8 \xC9\xB9 i\xCB\x90",
        "'three' -> θ ɹ iː");
  CHECK(r.words.size() == 1 && r.words[0] == "three", "'three' one word");
  CHECK(r.word_start.size() == 1 && r.word_start[0] == 0, "'three' word_start [0]");

  taver::g2p::G2PResult r2;
  CHECK(g2p.convert("three apples", r2, err), "convert 'three apples'");
  CHECK(r2.word_start.size() == r2.words.size(), "word arrays same length");
  CHECK(r2.words.size() == 2, "'three apples' two words");
  CHECK(r2.word_start.size() == 2 && r2.word_start[1] > 0, "second word starts later");

  // All-caps must not be spelled out as an initialism (lowercasing fix).
  taver::g2p::G2PResult r3;
  CHECK(g2p.convert("IT", r3, err), "convert 'IT'");
  CHECK(r3.phoneme_ids.size() <= 3, "'IT' is short (not spelled out)");

  if (failures == 0) std::printf("test_g2p: all checks passed\n");
  return failures ? 1 : 0;
}
