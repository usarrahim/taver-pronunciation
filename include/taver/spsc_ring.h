// Lock-free single-producer / single-consumer ring buffer.
//
// The audio callback thread (producer) and the DSP thread (consumer) hand data
// across this structure without ever taking a lock or allocating. Capacity is
// fixed at construction; the backing storage is a single contiguous block.
//
// Correctness relies on the classic acquire/release discipline:
//   * producer publishes by storing the new write index with release ordering,
//   * consumer reads the write index with acquire ordering (and vice-versa).
// One slot is intentionally left empty to disambiguate full vs. empty.
#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <vector>

namespace taver {

#if defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLine = 64;
#endif

template <typename T>
class SpscRing {
  static_assert(std::is_trivially_copyable_v<T>,
                "SpscRing is designed for trivially-copyable payloads");

 public:
  // `capacity` is the number of usable elements; one extra slot is reserved.
  explicit SpscRing(std::size_t capacity)
      : capacity_(capacity + 1), storage_(capacity + 1) {}

  SpscRing(const SpscRing&) = delete;
  SpscRing& operator=(const SpscRing&) = delete;

  // Producer side. Returns false if the buffer is full (no overwrite).
  bool push(const T& item) {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    const std::size_t next = increment(head);
    if (next == tail_.load(std::memory_order_acquire)) {
      return false;  // full
    }
    storage_[head] = item;
    head_.store(next, std::memory_order_release);
    return true;
  }

  // Consumer side. Returns false if the buffer is empty.
  bool pop(T& out) {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) {
      return false;  // empty
    }
    out = storage_[tail];
    tail_.store(increment(tail), std::memory_order_release);
    return true;
  }

  bool empty() const {
    return head_.load(std::memory_order_acquire) ==
           tail_.load(std::memory_order_acquire);
  }

  std::size_t size_approx() const {
    const std::size_t h = head_.load(std::memory_order_acquire);
    const std::size_t t = tail_.load(std::memory_order_acquire);
    return (h >= t) ? (h - t) : (capacity_ - t + h);
  }

  std::size_t capacity() const { return capacity_ - 1; }

 private:
  std::size_t increment(std::size_t idx) const {
    return (idx + 1) == capacity_ ? 0 : idx + 1;
  }

  const std::size_t capacity_;
  std::vector<T> storage_;

  // Keep the two indices on separate cache lines to avoid false sharing
  // between the producer and consumer cores.
  alignas(kCacheLine) std::atomic<std::size_t> head_{0};  // producer writes
  alignas(kCacheLine) std::atomic<std::size_t> tail_{0};  // consumer writes
};

}  // namespace taver
