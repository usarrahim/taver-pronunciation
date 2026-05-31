// Live microphone capture via WASAPI (Windows Audio Session API), event-driven
// shared mode for minimal capture latency. The device mix format is converted
// to mono 16 kHz on the capture thread and delivered in AudioChunk units.
#pragma once

#include <atomic>
#include <thread>
#include <vector>

#include "audio/audio_source.h"
#include "audio/resampler.h"

namespace taver::audio {

class WasapiSource : public IAudioSource {
 public:
  WasapiSource();
  ~WasapiSource() override;

  bool start(ChunkSink sink) override;
  void stop() override;
  bool finished() const override { return false; }  // microphone never ends

  const char* last_error() const { return last_error_; }

 private:
  void run();

  ChunkSink sink_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
  const char* last_error_ = "";
};

}  // namespace taver::audio
