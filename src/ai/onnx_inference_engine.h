// ONNX Runtime implementation of IInferenceEngine for wav2vec2-style phoneme
// CTC models. Input is the raw normalized 16 kHz waveform; output is per-frame
// phoneme log-probabilities.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ai/inference_engine.h"

namespace Ort {
struct Env;
struct Session;
struct MemoryInfo;
}  // namespace Ort

namespace taver::ai {

class OnnxInferenceEngine : public IInferenceEngine {
 public:
  OnnxInferenceEngine();
  ~OnnxInferenceEngine() override;

  bool load(const std::string& model_path, Backend backend,
            std::string& err) override;
  int vocab_size() const override { return vocab_size_; }
  const char* provider() const override { return provider_.c_str(); }
  bool infer(const float* samples, int num_samples, LogitMatrix& out) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int vocab_size_ = 0;
  std::string provider_ = "none";

  // Reusable input buffer (normalized waveform). Resized once to the max.
  std::vector<float> norm_;
};

}  // namespace taver::ai
