#ifndef INCLUDE_DATA_COMPRESSOR
#define INCLUDE_DATA_COMPRESSOR

#include <atomic>
#include <chrono>
#include <string>
#include <sys/types.h>
#include <vector>

class DataCompressor {
private:
  std::vector<std::atomic<bool>> _threads_busy;
  std::chrono::milliseconds _sleep_duration;

public:
  DataCompressor(u_int32_t thread_max, int sleep_duration_ms);
  // non blocking if thread available, otherwise waits for free one
  void save_memcpy_data(void *data, size_t size, std::string name);
  u_int32_t busy_threads();
};

#endif
