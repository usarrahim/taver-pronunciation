// Fixed-capacity object pool.
//
// Engineering rule for this project: no new/malloc on the data path while the
// pipeline is running. Large working buffers (utterance audio, feature
// matrices, logit tensors) are acquired from a pool that is fully pre-allocated
// at startup. Acquiring/releasing a block is an O(1) index push/pop and never
// touches the system allocator.
//
// This pool is used across thread boundaries (DSP thread acquires, inference
// thread releases) so the free-list is guarded by a tiny spinlock. The lock is
// held only for the duration of an index swap — microseconds — and is never
// held across real work, so it does not violate the "don't block the pipeline"
// constraint.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace taver {

template <typename T>
class MemoryPool {
 public:
  explicit MemoryPool(std::size_t count) : blocks_(count) {
    free_list_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      free_list_.push_back(&blocks_[i]);
    }
  }

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  // Returns nullptr when exhausted; callers must handle back-pressure rather
  // than allocating.
  T* acquire() {
    Guard g(flag_);
    if (free_list_.empty()) return nullptr;
    T* p = free_list_.back();
    free_list_.pop_back();
    return p;
  }

  void release(T* p) {
    if (!p) return;
    Guard g(flag_);
    free_list_.push_back(p);
  }

  std::size_t available() const {
    Guard g(flag_);
    return free_list_.size();
  }

  std::size_t capacity() const { return blocks_.size(); }

 private:
  // Minimal RAII spinlock; uncontended in practice (few blocks in flight).
  struct Guard {
    explicit Guard(std::atomic_flag& f) : f_(f) {
      while (f_.test_and_set(std::memory_order_acquire)) {
      }
    }
    ~Guard() { f_.clear(std::memory_order_release); }
    std::atomic_flag& f_;
  };

  std::vector<T> blocks_;
  std::vector<T*> free_list_;
  mutable std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

}  // namespace taver
