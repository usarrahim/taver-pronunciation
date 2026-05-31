#include "audio/wasapi_source.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <avrt.h>
#include <functiondiscoverykeys_devpkey.h>

#include <chrono>
#include <cstring>

#include "taver/config.h"

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

namespace taver::audio {

namespace {
std::uint64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

template <typename T>
void safe_release(T*& p) {
  if (p) {
    p->Release();
    p = nullptr;
  }
}

bool is_float_format(const WAVEFORMATEX* wf) {
  if (wf->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) return true;
  if (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
    const auto* ext = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wf);
    return ext->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
  }
  return false;
}
}  // namespace

WasapiSource::WasapiSource() = default;
WasapiSource::~WasapiSource() { stop(); }

bool WasapiSource::start(ChunkSink sink) {
  sink_ = std::move(sink);
  stop_ = false;
  thread_ = std::thread([this] { run(); });
  return true;
}

void WasapiSource::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
}

void WasapiSource::run() {
  HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  const bool com_inited = SUCCEEDED(hr);

  IMMDeviceEnumerator* enumerator = nullptr;
  IMMDevice* device = nullptr;
  IAudioClient* client = nullptr;
  IAudioCaptureClient* capture = nullptr;
  WAVEFORMATEX* mix = nullptr;
  HANDLE audio_event = nullptr;
  HANDLE mmcss = nullptr;

  auto fail = [&](const char* msg) {
    last_error_ = msg;
    safe_release(capture);
    safe_release(client);
    safe_release(device);
    safe_release(enumerator);
    if (mix) CoTaskMemFree(mix);
    if (audio_event) CloseHandle(audio_event);
    if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
    if (com_inited) CoUninitialize();
  };

  hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                        __uuidof(IMMDeviceEnumerator),
                        reinterpret_cast<void**>(&enumerator));
  if (FAILED(hr)) return fail("CoCreateInstance(MMDeviceEnumerator) failed");

  hr = enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &device);
  if (FAILED(hr)) return fail("No default capture device");

  hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                        reinterpret_cast<void**>(&client));
  if (FAILED(hr)) return fail("IAudioClient activate failed");

  hr = client->GetMixFormat(&mix);
  if (FAILED(hr)) return fail("GetMixFormat failed");

  // ~30 ms shared buffer; event-driven so we wake exactly when data is ready.
  const REFERENCE_TIME buf_dur = 300000;  // 30 ms in 100-ns units
  hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                          AUDCLNT_STREAMFLAGS_EVENTCALLBACK, buf_dur, 0, mix,
                          nullptr);
  if (FAILED(hr)) return fail("IAudioClient::Initialize failed");

  audio_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!audio_event) return fail("CreateEvent failed");
  hr = client->SetEventHandle(audio_event);
  if (FAILED(hr)) return fail("SetEventHandle failed");

  hr = client->GetService(__uuidof(IAudioCaptureClient),
                          reinterpret_cast<void**>(&capture));
  if (FAILED(hr)) return fail("GetService(IAudioCaptureClient) failed");

  const int in_rate = static_cast<int>(mix->nSamplesPerSec);
  const int in_ch = mix->nChannels;
  const bool is_float = is_float_format(mix);
  const int bits = mix->wBitsPerSample;
  Resampler resampler(in_rate, in_ch, kTargetSampleRate);

  // Promote this thread to Pro Audio scheduling for stable low latency.
  DWORD task_index = 0;
  mmcss = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task_index);

  hr = client->Start();
  if (FAILED(hr)) return fail("IAudioClient::Start failed");

  std::vector<float> mono;
  mono.reserve(4096);
  AudioChunk chunk;
  std::uint32_t fill = 0;

  while (!stop_.load()) {
    const DWORD wait = WaitForSingleObject(audio_event, 200);
    if (wait != WAIT_OBJECT_0) continue;  // timeout -> re-check stop flag

    UINT32 packet = 0;
    capture->GetNextPacketSize(&packet);
    while (packet != 0 && !stop_.load()) {
      BYTE* data = nullptr;
      UINT32 frames = 0;
      DWORD flags = 0;
      hr = capture->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
      if (FAILED(hr)) break;

      mono.clear();
      if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
        static thread_local std::vector<float> zeros;
        zeros.assign(static_cast<std::size_t>(frames) * in_ch, 0.0f);
        resampler.process(zeros.data(), frames, mono);
      } else if (is_float && bits == 32) {
        resampler.process(reinterpret_cast<const float*>(data), frames, mono);
      } else if (bits == 16) {
        static thread_local std::vector<float> tmp;
        const auto* pcm = reinterpret_cast<const std::int16_t*>(data);
        tmp.resize(static_cast<std::size_t>(frames) * in_ch);
        for (std::size_t i = 0; i < tmp.size(); ++i) tmp[i] = pcm[i] / 32768.0f;
        resampler.process(tmp.data(), frames, mono);
      } else if (bits == 32) {  // int32 PCM
        static thread_local std::vector<float> tmp;
        const auto* pcm = reinterpret_cast<const std::int32_t*>(data);
        tmp.resize(static_cast<std::size_t>(frames) * in_ch);
        for (std::size_t i = 0; i < tmp.size(); ++i)
          tmp[i] = pcm[i] / 2147483648.0f;
        resampler.process(tmp.data(), frames, mono);
      }
      capture->ReleaseBuffer(frames);

      for (float s : mono) {
        chunk.samples[fill++] = s;
        if (fill == kCaptureChunkSamples) {
          chunk.count = fill;
          chunk.timestamp_ns = now_ns();
          if (sink_) sink_(chunk);
          fill = 0;
        }
      }
      capture->GetNextPacketSize(&packet);
    }
  }

  client->Stop();
  fail("");  // reuse cleanup path (clears error to empty)
}

}  // namespace taver::audio
