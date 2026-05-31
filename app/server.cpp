// Taver local web server.
//
// A tiny, dependency-free HTTP/1.1 server (WinSock) that hosts the browser UI
// and bridges the browser microphone to the C++ engine. The browser captures
// audio and POSTs a WAV; we score it with the exact same Taver.dll the CLI
// uses, and return JSON. This is the Phase-4 "use the library as an API" path.
//
//   GET  /                -> web/index.html
//   GET  /<file>          -> static file from --root
//   POST /score?target=.. -> body is a WAV blob; returns JSON pronunciation result
//
// Single-threaded on purpose: scoring takes ~150 ms and one learner speaks at a
// time, so a simple accept->serve->close loop keeps the code auditable.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include "taver/taver.h"

#pragma comment(lib, "ws2_32.lib")

namespace {

std::string g_root = "web";
TaverEngine* g_engine = nullptr;

std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
      };
      out.push_back(static_cast<char>(hex(s[i + 1]) * 16 + hex(s[i + 2])));
      i += 2;
    } else if (s[i] == '+') {
      out.push_back(' ');
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

const char* content_type(const std::string& path) {
  auto ends = [&](const char* e) {
    const std::size_t n = std::strlen(e);
    return path.size() >= n && path.compare(path.size() - n, n, e) == 0;
  };
  if (ends(".html")) return "text/html; charset=utf-8";
  if (ends(".js")) return "application/javascript; charset=utf-8";
  if (ends(".css")) return "text/css; charset=utf-8";
  if (ends(".svg")) return "image/svg+xml";
  return "application/octet-stream";
}

void send_all(SOCKET c, const char* data, int len) {
  int sent = 0;
  while (sent < len) {
    const int n = send(c, data + sent, len - sent, 0);
    if (n <= 0) break;
    sent += n;
  }
}

void respond(SOCKET c, int code, const char* status, const std::string& ctype,
             const std::string& body) {
  std::ostringstream h;
  h << "HTTP/1.1 " << code << ' ' << status << "\r\n"
    << "Content-Type: " << ctype << "\r\n"
    << "Content-Length: " << body.size() << "\r\n"
    << "Access-Control-Allow-Origin: *\r\n"
    << "Connection: close\r\n\r\n";
  const std::string head = h.str();
  send_all(c, head.data(), static_cast<int>(head.size()));
  send_all(c, body.data(), static_cast<int>(body.size()));
}

void serve_file(SOCKET c, std::string path) {
  if (path == "/" || path.empty()) path = "/index.html";
  // Prevent directory traversal.
  if (path.find("..") != std::string::npos) {
    respond(c, 400, "Bad Request", "text/plain", "bad path");
    return;
  }
  const std::string full = g_root + path;
  std::ifstream f(full, std::ios::binary);
  if (!f) {
    respond(c, 404, "Not Found", "text/plain", "not found: " + path);
    return;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  respond(c, 200, "OK", content_type(path), ss.str());
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') { o.push_back('\\'); o.push_back(ch); }
    else if (ch == '\n') o += "\\n";
    else o.push_back(ch);
  }
  return o;
}

const char* verdict_name(int e) {
  switch (e) {
    case TAVER_PHONE_GOOD: return "good";
    case TAVER_PHONE_FAIR: return "fair";
    case TAVER_PHONE_SUBSTITUTED: return "substituted";
    case TAVER_PHONE_DELETED: return "deleted";
    case TAVER_PHONE_INSERTED: return "inserted";
    default: return "unknown";
  }
}

std::string result_to_json(const TaverResult& r) {
  std::ostringstream j;
  j << "{\"overall\":" << r.overall_score
    << ",\"utterance_sec\":" << r.utterance_sec
    << ",\"latency_ms\":" << r.latency_ms
    << ",\"speaking_rate_pps\":" << r.speaking_rate_pps
    << ",\"mean_pitch_hz\":" << r.mean_pitch_hz
    << ",\"pitch_range_hz\":" << r.pitch_range_hz
    << ",\"fluency_score\":" << r.fluency_score
    << ",\"words\":[";
  for (int i = 0; i < r.num_words; ++i) {
    const auto& w = r.words[i];
    if (i) j << ',';
    j << "{\"text\":\"" << json_escape(w.text) << "\""
      << ",\"score\":" << w.score
      << ",\"phoneme_start\":" << w.phoneme_start
      << ",\"phoneme_count\":" << w.phoneme_count
      << ",\"has_error\":" << (w.has_error ? "true" : "false") << "}";
  }
  j << "],\"phonemes\":[";
  for (int i = 0; i < r.num_phonemes; ++i) {
    const auto& p = r.phonemes[i];
    if (i) j << ',';
    j << "{\"expected\":\"" << json_escape(p.phoneme) << "\""
      << ",\"realized\":\"" << json_escape(p.realized) << "\""
      << ",\"score\":" << p.score
      << ",\"verdict\":\"" << verdict_name(p.error_type) << "\""
      << ",\"start\":" << p.start_sec
      << ",\"duration\":" << p.duration_sec << "}";
  }
  j << "]}";
  return j.str();
}

// Extract a single url-encoded query parameter value.
std::string query_param(const std::string& query, const std::string& key) {
  const std::string probe = key + "=";
  auto pos = query.find(probe);
  if (pos == std::string::npos) return "";
  std::string raw = query.substr(pos + probe.size());
  const auto amp = raw.find('&');
  if (amp != std::string::npos) raw = raw.substr(0, amp);
  return url_decode(raw);
}

// Set the engine target from either free text (preferred) or a phoneme string.
TaverStatus apply_target(const std::string& query) {
  const std::string voice = query_param(query, "voice");
  if (voice == "en-gb" || voice == "gb") taver_set_g2p_voice(g_engine, 1);
  else if (!voice.empty()) taver_set_g2p_voice(g_engine, 0);

  const std::string text = query_param(query, "text");
  if (!text.empty()) return taver_set_target_text(g_engine, text.c_str());
  const std::string target = query_param(query, "target");
  if (!target.empty()) return taver_set_target(g_engine, target.c_str());
  return TAVER_ERR_NO_TARGET;
}

// GET /g2p?text=... -> resolved phoneme string + IPA (preview before recording).
void handle_g2p(SOCKET c, const std::string& query) {
  if (apply_target(query) != TAVER_OK) {
    respond(c, 200, "OK", "application/json",
            "{\"error\":\"could not phonemize text\"}");
    return;
  }
  char ph[2048] = {0}, ipa[2048] = {0};
  taver_get_target_info(g_engine, ph, sizeof(ph), ipa, sizeof(ipa));
  std::ostringstream j;
  j << "{\"phonemes\":\"" << json_escape(ph) << "\",\"ipa\":\"" << json_escape(ipa) << "\"}";
  respond(c, 200, "OK", "application/json", j.str());
}

void handle_score(SOCKET c, const std::string& query, const std::string& body) {
  // Target comes from ?text=<words/sentence> (G2P) or ?target=<phonemes>.
  TaverStatus st = apply_target(query);
  if (st == TAVER_ERR_NO_TARGET) {
    respond(c, 400, "Bad Request", "application/json",
            "{\"error\":\"missing target (text or phonemes)\"}");
    return;
  }
  if (st != TAVER_OK) {
    respond(c, 400, "Bad Request", "application/json",
            std::string("{\"error\":\"") + taver_status_string(st) + "\"}");
    return;
  }
  if (body.size() < 44) {
    respond(c, 400, "Bad Request", "application/json",
            "{\"error\":\"empty or invalid audio\"}");
    return;
  }

  // Persist the uploaded WAV to a temp file and score it with the engine.
  char tmpdir[MAX_PATH];
  GetTempPathA(MAX_PATH, tmpdir);
  std::string tmp = std::string(tmpdir) + "taver_rec.wav";
  {
    std::ofstream out(tmp, std::ios::binary);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
  }

  TaverResult r{};
  st = taver_score_wav(g_engine, tmp.c_str(), &r);
  if (st != TAVER_OK) {
    respond(c, 200, "OK", "application/json",
            std::string("{\"error\":\"") + taver_status_string(st) +
                "\",\"detail\":\"no speech detected? try again, louder/closer\"}");
    return;
  }
  respond(c, 200, "OK", "application/json", result_to_json(r));
}

bool read_request(SOCKET c, std::string& method, std::string& path,
                  std::string& query, std::string& body) {
  std::string buf;
  char tmp[8192];
  // Read until end of headers.
  std::size_t header_end = std::string::npos;
  while (true) {
    const int n = recv(c, tmp, sizeof(tmp), 0);
    if (n <= 0) return false;
    buf.append(tmp, n);
    header_end = buf.find("\r\n\r\n");
    if (header_end != std::string::npos) break;
    if (buf.size() > 1u << 20) return false;  // 1 MB header cap
  }

  // Request line.
  const std::size_t line_end = buf.find("\r\n");
  std::istringstream rl(buf.substr(0, line_end));
  std::string url;
  rl >> method >> url;
  const auto q = url.find('?');
  if (q != std::string::npos) { path = url.substr(0, q); query = url.substr(q + 1); }
  else { path = url; query.clear(); }

  // Content-Length.
  std::size_t content_len = 0;
  {
    const std::string headers = buf.substr(0, header_end);
    std::string lower = headers;
    for (auto& ch : lower) ch = static_cast<char>(tolower(ch));
    const auto cl = lower.find("content-length:");
    if (cl != std::string::npos) content_len = std::strtoul(headers.c_str() + cl + 15, nullptr, 10);
  }

  // Body (read remainder up to content_len).
  body = buf.substr(header_end + 4);
  while (body.size() < content_len) {
    const int n = recv(c, tmp, sizeof(tmp), 0);
    if (n <= 0) break;
    body.append(tmp, n);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model = "models/phoneme.int8.onnx";
  std::string vocab = "models/phoneme_vocab.txt";
  std::string espeak = "third_party/espeak/eSpeak NG/espeak-ng.exe";
  std::string host = "127.0.0.1";
  int port = 8080;
  int backend = TAVER_BACKEND_AUTO;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* d) { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(d); };
    if (a == "--model") model = next(model.c_str());
    else if (a == "--vocab") vocab = next(vocab.c_str());
    else if (a == "--espeak") espeak = next(espeak.c_str());
    else if (a == "--root") g_root = next(g_root.c_str());
    else if (a == "--port") port = std::atoi(next("8080").c_str());
    else if (a == "--host") host = next("127.0.0.1");
    else if (a == "--backend") { std::string b = next("auto");
      backend = (b == "cpu") ? TAVER_BACKEND_CPU : (b == "cuda") ? TAVER_BACKEND_CUDA
              : (b == "dml" || b == "directml") ? TAVER_BACKEND_DIRECTML : TAVER_BACKEND_AUTO; }
  }

  std::printf("%s - loading model %s ...\n", taver_version(), model.c_str());
  TaverConfig cfg{};
  cfg.model_path = model.c_str();
  cfg.vocab_path = vocab.c_str();
  cfg.espeak_path = espeak.c_str();
  cfg.lexicon_path = "data/lexicon.txt";
  cfg.calibration_path = "config/calibration.json";
  cfg.backend = backend;
  cfg.g2p_voice = 0;
  TaverStatus st = taver_create(&cfg, &g_engine);
  if (st != TAVER_OK) {
    std::printf("Engine init failed: %s\n", taver_status_string(st));
    return 1;
  }

  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { std::printf("WSAStartup failed\n"); return 1; }
  SOCKET srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  BOOL yes = TRUE;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&yes), sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(srv, 16) != 0) {
    std::printf("Cannot bind to %s:%d\n", host.c_str(), port);
    return 1;
  }

  std::printf("\n  Taver UI ready ->  http://%s:%d\n", host.c_str(), port);
  std::printf("  (open it in your browser, allow the microphone, pick a word, hold to record)\n\n");

  for (;;) {
    SOCKET c = accept(srv, nullptr, nullptr);
    if (c == INVALID_SOCKET) continue;
    std::string method, path, query, body;
    if (read_request(c, method, path, query, body)) {
      if (method == "POST" && path == "/score") handle_score(c, query, body);
      else if (method == "GET" && path == "/g2p") handle_g2p(c, query);
      else if (method == "GET") serve_file(c, path);
      else respond(c, 405, "Method Not Allowed", "text/plain", "no");
    }
    closesocket(c);
  }
}
