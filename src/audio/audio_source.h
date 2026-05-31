// Abstract audio input. Both the live WASAPI microphone and the offline WAV
// reader implement this, so the rest of the pipeline is agnostic to where the
// samples come from. Sources always deliver mono float32 already resampled to
// kTargetSampleRate, in fixed AudioChunk units.
#pragma once

#include <functional>

#include "taver/types.h"

namespace taver::audio {

class IAudioSource {
 public:
  using ChunkSink = std::function<void(const AudioChunk&)>;

  virtual ~IAudioSource() = default;

  // Begin capture. `sink` is invoked from the source's own thread for each
  // chunk of kCaptureChunkSamples (or fewer at end-of-stream). Returns false on
  // device/file error.
  virtual bool start(ChunkSink sink) = 0;

  // Stop capture and join the capture thread.
  virtual void stop() = 0;

  // True once a finite source (file) has delivered all of its audio.
  virtual bool finished() const = 0;
};

}  // namespace taver::audio
