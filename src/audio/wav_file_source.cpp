#include "audio/wav_file_source.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>

#include "taver/config.h"

namespace taver::audio {

namespace {
std::uint32_t rd_u32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint16_t rd_u16(const unsigned char* p) {
  return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}
std::uint64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

WavFileSource::WavFileSource(std::string path) : path_(std::move(path)) {
  loaded_ = load();
}

WavFileSource::~WavFileSource() { stop(); }

bool WavFileSource::load() {
  std::ifstream f(path_, std::ios::binary);
  if (!f) return false;
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
  if (buf.size() < 44) return false;
  if (std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
    return false;
  }

  std::uint16_t fmt = 0, channels = 0, bits = 0;
  std::uint32_t rate = 0;
  const unsigned char* data = nullptr;
  std::uint32_t data_len = 0;

  std::size_t pos = 12;
  while (pos + 8 <= buf.size()) {
    const unsigned char* ck = buf.data() + pos;
    const std::uint32_t id = rd_u32(ck);  // little-endian fourcc
    const std::uint32_t sz = rd_u32(ck + 4);
    const unsigned char* body = ck + 8;
    if (std::memcmp(ck, "fmt ", 4) == 0 && sz >= 16) {
      fmt = rd_u16(body);
      channels = rd_u16(body + 2);
      rate = rd_u32(body + 4);
      bits = rd_u16(body + 14);
    } else if (std::memcmp(ck, "data", 4) == 0) {
      data = body;
      data_len = sz;
      if (body + sz > buf.data() + buf.size())
        data_len = static_cast<std::uint32_t>(buf.data() + buf.size() - body);
    }
    (void)id;
    pos += 8 + sz + (sz & 1);  // chunks are word-aligned
  }

  if (!data || channels == 0 || rate == 0) return false;

  // Decode interleaved samples to float [-1, 1].
  std::vector<float> interleaved;
  const std::uint32_t bytes_per_sample = bits / 8;
  if (bytes_per_sample == 0) return false;
  const std::uint32_t total = data_len / bytes_per_sample;
  interleaved.reserve(total);

  if (fmt == 3 && bits == 32) {  // IEEE float
    for (std::uint32_t i = 0; i < total; ++i) {
      float v;
      std::memcpy(&v, data + static_cast<std::size_t>(i) * 4, 4);
      interleaved.push_back(v);
    }
  } else if (fmt == 1 || fmt == 0xFFFE) {  // PCM (or extensible)
    for (std::uint32_t i = 0; i < total; ++i) {
      const unsigned char* s = data + static_cast<std::size_t>(i) * bytes_per_sample;
      float v = 0.0f;
      if (bits == 16) {
        v = static_cast<std::int16_t>(rd_u16(s)) / 32768.0f;
      } else if (bits == 24) {
        std::int32_t x = (s[0] | (s[1] << 8) | (s[2] << 16));
        if (x & 0x800000) x |= ~0xFFFFFF;  // sign-extend
        v = x / 8388608.0f;
      } else if (bits == 32) {
        v = static_cast<std::int32_t>(rd_u32(s)) / 2147483648.0f;
      } else if (bits == 8) {
        v = (static_cast<int>(s[0]) - 128) / 128.0f;  // 8-bit PCM is unsigned
      } else {
        return false;
      }
      interleaved.push_back(v);
    }
  } else {
    return false;
  }

  Resampler rs(static_cast<int>(rate), channels, kTargetSampleRate);
  const int frames = static_cast<int>(interleaved.size() / channels);
  rs.process(interleaved.data(), frames, mono_);
  rs.flush(mono_);
  return !mono_.empty();
}

bool WavFileSource::start(ChunkSink sink) {
  if (!loaded_) return false;
  sink_ = std::move(sink);
  finished_ = false;
  stop_ = false;
  thread_ = std::thread([this] { run(); });
  return true;
}

void WavFileSource::run() {
  AudioChunk chunk;
  std::size_t i = 0;
  while (i < mono_.size() && !stop_.load()) {
    std::uint32_t n = 0;
    while (n < kCaptureChunkSamples && i < mono_.size()) {
      chunk.samples[n++] = mono_[i++];
    }
    chunk.count = n;
    chunk.timestamp_ns = now_ns();
    if (sink_) sink_(chunk);
  }
  finished_ = true;
}

void WavFileSource::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

}  // namespace taver::audio
