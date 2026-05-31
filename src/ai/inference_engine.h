// Abstract acoustic-model inference. Decouples the pipeline from ONNX Runtime
// so an alternative backend (TFLite, a custom kernel, a mock for tests) can be
// dropped in without touching the alignment or audio layers.
#pragma once

#include <string>

#include "taver/types.h"

namespace taver::ai {

enum class Backend { kAuto, kCpu, kCuda, kDirectML };

class IInferenceEngine {
 public:
  virtual ~IInferenceEngine() = default;

  // Load a model. On failure returns false and fills `err`.
  virtual bool load(const std::string& model_path, Backend backend,
                    std::string& err) = 0;

  // Number of output classes (phoneme vocabulary size).
  virtual int vocab_size() const = 0;

  // Human-readable description of the active execution provider.
  virtual const char* provider() const = 0;

  // Run the acoustic model on raw mono 16 kHz audio. Frame-level log-probabilities
  // are written into `out` (num_frames x vocab_size). Returns false on error.
  // Must not allocate large buffers on the hot path beyond what ORT requires.
  virtual bool infer(const float* samples, int num_samples, LogitMatrix& out) = 0;
};

}  // namespace taver::ai
