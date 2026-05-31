#include "ai/onnx_inference_engine.h"

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "taver/config.h"

namespace taver::ai {

struct OnnxInferenceEngine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "Taver"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions alloc;
  std::string input_name;
  std::string output_name;
};

OnnxInferenceEngine::OnnxInferenceEngine() : impl_(std::make_unique<Impl>()) {}
OnnxInferenceEngine::~OnnxInferenceEngine() = default;

namespace {
std::wstring widen(const std::string& s) {
  return std::wstring(s.begin(), s.end());
}
}  // namespace

bool OnnxInferenceEngine::load(const std::string& model_path, Backend backend,
                               std::string& err) {
  try {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(0);  // 0 => ORT picks a sensible default
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    provider_ = "CPU";
    const auto avail = Ort::GetAvailableProviders();
    auto has = [&](const char* name) {
      for (const auto& p : avail) if (p == name) return true;
      return false;
    };

    // CUDA (NVIDIA): only attempt it if this ONNX Runtime build ships the
    // provider, so CPU-only machines never log a scary error.
    if (backend == Backend::kAuto || backend == Backend::kCuda) {
      if (has("CUDAExecutionProvider")) {
        try {
          OrtCUDAProviderOptions cuda{};
          cuda.device_id = 0;
          opts.AppendExecutionProvider_CUDA(cuda);
          provider_ = "CUDA";
        } catch (const Ort::Exception&) { provider_ = "CPU"; }
      } else if (backend == Backend::kCuda) {
        err = "CUDA requested but this ONNX Runtime build is CPU-only; using CPU";
      }
    }

    // DirectML (any Direct3D 12 GPU on Windows). Resolved at runtime from the
    // loaded onnxruntime.dll so we keep zero compile/link dependency on the DML
    // package: drop in an onnxruntime-directml build and this lights up.
    if (provider_ == "CPU" && (backend == Backend::kAuto || backend == Backend::kDirectML)) {
      if (has("DmlExecutionProvider")) {
        using DmlAppend = OrtStatus* (*)(OrtSessionOptions*, int);
        HMODULE ort = GetModuleHandleW(L"onnxruntime.dll");
        auto fn = ort ? reinterpret_cast<DmlAppend>(
                            GetProcAddress(ort, "OrtSessionOptionsAppendExecutionProvider_DML"))
                      : nullptr;
        if (fn) {
          // DML requires sequential execution with memory pattern disabled.
          opts.DisableMemPattern();
          opts.SetExecutionMode(ORT_SEQUENTIAL);
          OrtStatus* s = fn(static_cast<OrtSessionOptions*>(opts), 0);
          if (s == nullptr) provider_ = "DirectML";
          else Ort::GetApi().ReleaseStatus(s);
        }
      } else if (backend == Backend::kDirectML) {
        err = "DirectML requested but this ONNX Runtime build lacks the DML provider; using CPU";
      }
    }

#if defined(_WIN32)
    const std::wstring wpath = widen(model_path);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(), opts);
#else
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), opts);
#endif

    // Cache I/O names.
    Ort::AllocatedStringPtr in = impl_->session->GetInputNameAllocated(0, impl_->alloc);
    Ort::AllocatedStringPtr out = impl_->session->GetOutputNameAllocated(0, impl_->alloc);
    impl_->input_name = in.get();
    impl_->output_name = out.get();

    // Vocab size = last dimension of the output tensor.
    auto out_info = impl_->session->GetOutputTypeInfo(0);
    auto shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
    if (!shape.empty() && shape.back() > 0) {
      vocab_size_ = static_cast<int>(shape.back());
    }
    norm_.resize(kMaxUtteranceSamples);
    return true;
  } catch (const Ort::Exception& e) {
    err = std::string("ONNX load failed: ") + e.what();
    return false;
  }
}

bool OnnxInferenceEngine::infer(const float* samples, int num_samples,
                                LogitMatrix& out) {
  if (!impl_->session || num_samples <= 0) return false;
  if (num_samples > kMaxUtteranceSamples) num_samples = kMaxUtteranceSamples;

  // wav2vec2 feature extractor normalizes each utterance to zero mean, unit var.
  double mean = 0.0;
  for (int i = 0; i < num_samples; ++i) mean += samples[i];
  mean /= num_samples;
  double var = 0.0;
  for (int i = 0; i < num_samples; ++i) {
    const double d = samples[i] - mean;
    var += d * d;
  }
  var /= num_samples;
  const float inv_std = static_cast<float>(1.0 / std::sqrt(var + 1e-7));
  for (int i = 0; i < num_samples; ++i) {
    norm_[i] = static_cast<float>((samples[i] - mean) * inv_std);
  }

  try {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t in_shape[2] = {1, num_samples};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, norm_.data(), static_cast<size_t>(num_samples), in_shape, 2);

    const char* in_names[] = {impl_->input_name.c_str()};
    const char* out_names[] = {impl_->output_name.c_str()};
    auto results = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                       out_names, 1);

    Ort::Value& logits = results.front();
    auto info = logits.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();  // [1, frames, vocab]
    if (shape.size() != 3) return false;
    const int frames = static_cast<int>(shape[1]);
    const int vocab = static_cast<int>(shape[2]);
    const float* data = logits.GetTensorData<float>();

    out.num_frames = std::min(frames, kMaxModelFrames);
    out.vocab_size = std::min(vocab, kMaxVocab);
    vocab_size_ = vocab;

    // Convert raw logits to log-probabilities (log-softmax) per frame. The GOP
    // scorer works in log-prob space.
    for (int f = 0; f < out.num_frames; ++f) {
      const float* src = data + static_cast<size_t>(f) * vocab;
      float* dst = out.row(f);
      float maxv = src[0];
      for (int v = 1; v < vocab; ++v) maxv = std::max(maxv, src[v]);
      float sum = 0.0f;
      for (int v = 0; v < out.vocab_size; ++v) sum += std::exp(src[v] - maxv);
      const float logsum = std::log(sum) + maxv;
      for (int v = 0; v < out.vocab_size; ++v) dst[v] = src[v] - logsum;
    }
    return true;
  } catch (const Ort::Exception&) {
    return false;
  }
}

}  // namespace taver::ai
