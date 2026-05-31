// Finite audio source backed by a WAV file. Decodes PCM (16/24/32-bit int or
// 32-bit float), down-mixes + resamples to mono 16 kHz, and streams the result
// in AudioChunk units on a worker thread, exactly like the live microphone.
#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "audio/audio_source.h"
#include "audio/resampler.h"

namespace taver::audio {

class WavFileSource : public IAudioSource {
 public:
  explicit WavFileSource(std::string path);
  ~WavFileSource() override;

  bool start(ChunkSink sink) override;
  void stop() override;
  bool finished() const override { return finished_.load(); }

  // Exposed for the synchronous scoring path.
  bool ok() const { return loaded_; }
  const std::vector<float>& mono() const { return mono_; }

 private:
  bool load();
  void run();

  std::string path_;
  ChunkSink sink_;
  std::vector<float> mono_;  // full decoded mono 16 kHz signal
  std::atomic<bool> finished_{false};
  std::atomic<bool> stop_{false};
  bool loaded_ = false;
  std::thread thread_;
};

}  // namespace taver::audio
