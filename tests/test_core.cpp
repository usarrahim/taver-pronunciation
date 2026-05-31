// Concurrency + correctness tests for the lock-free SPSC ring and memory pool.
#include <atomic>
#include <cstdio>
#include <thread>

#include "taver/spsc_ring.h"
#include "taver/memory_pool.h"

static int failures = 0;
#define CHECK(cond, msg)                                  \
  do {                                                    \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } \
  } while (0)

static void test_spsc_basic() {
  taver::SpscRing<int> q(4);
  CHECK(q.empty(), "new ring empty");
  CHECK(q.push(1) && q.push(2) && q.push(3) && q.push(4), "fill to capacity");
  CHECK(!q.push(5), "push rejected when full");
  int v = 0;
  CHECK(q.pop(v) && v == 1, "fifo order 1");
  CHECK(q.pop(v) && v == 2, "fifo order 2");
  CHECK(q.push(5), "push after pop");
}

static void test_spsc_threaded() {
  constexpr int N = 200000;
  taver::SpscRing<int> q(1024);
  std::atomic<bool> ok{true};
  std::thread producer([&] {
    for (int i = 0; i < N; ++i)
      while (!q.push(i)) std::this_thread::yield();
  });
  long long sum = 0;
  int got = 0, expect = 0;
  while (got < N) {
    int v;
    if (q.pop(v)) {
      if (v != expect) ok = false;  // strict FIFO across threads
      ++expect;
      sum += v;
      ++got;
    }
  }
  producer.join();
  CHECK(ok.load(), "spsc preserves order under contention");
  CHECK(sum == (long long)(N - 1) * N / 2, "spsc no lost/dup items");
}

static void test_pool() {
  taver::MemoryPool<int> pool(3);
  CHECK(pool.available() == 3, "pool starts full");
  int* a = pool.acquire();
  int* b = pool.acquire();
  int* c = pool.acquire();
  CHECK(a && b && c, "acquire 3");
  CHECK(pool.acquire() == nullptr, "exhausted pool returns null (no alloc)");
  pool.release(b);
  CHECK(pool.available() == 1, "release returns block");
  CHECK(pool.acquire() == b, "reuse released block");
  pool.release(a);
  pool.release(c);
}

int main() {
  test_spsc_basic();
  test_spsc_threaded();
  test_pool();
  if (failures == 0) std::printf("test_core: ALL PASS\n");
  return failures == 0 ? 0 : 1;
}
