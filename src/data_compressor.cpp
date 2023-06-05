#include "./data_compressor.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/types.h>
#include <thread>
#include <zip.h>

void compress_to_file(void *data, size_t size, std::string name) {
  zip_t *archive = zip_open(name.c_str(), ZIP_CREATE | ZIP_TRUNCATE, NULL);
  if (!archive) {
    std::cout << "[ERROR]: ziplib could not create archive for file: " << name
              << std::endl;
  }

  zip_error_t error;
  zip_source_t *source = zip_source_buffer_create(data, size, 0, &error);
  if (!source) {
    std::cout << "[ERROR]: ziplib " << zip_error_strerror(&error) << std::endl;
  }

  if (zip_file_add(archive, name.c_str(), source, ZIP_FL_ENC_UTF_8) < 0) {
    std::cout << "[ERROR]: ziplib could not add data for file: " << name
              << std::endl;
  }

  zip_close(archive);
}

DataCompressor::DataCompressor(u_int32_t thread_max, int sleep_duration_ms) {
  _threads_busy = std::vector<std::atomic<bool>>(thread_max);
  for (auto &t : _threads_busy)
    t = false;
  _sleep_duration = std::chrono::milliseconds(sleep_duration_ms);
}

void DataCompressor::save_memcpy_data(void *data, size_t size,
                                      std::string name) {
  void *data_copied = malloc(size);
  memcpy(data_copied, data, size);

  while (true) {
    for (u_int32_t i = 0; i < _threads_busy.size(); i++) {
      if (_threads_busy[i] == true)
        continue;

      _threads_busy[i] = true;
      std::thread t([this, i, data_copied, size, name] {
        compress_to_file(data_copied, size, name);
        free(data_copied);
        _threads_busy[i] = false;
      });
      t.detach();
      return;
    }
    std::this_thread::sleep_for(_sleep_duration);
  }
}

u_int32_t DataCompressor::busy_threads() {
  u_int32_t busy = 0;
  for (auto &t : _threads_busy)
    if (t)
      busy++;
  return busy;
}

void DataCompressor::join() {
  size_t i = 0;
  while (i < _threads_busy.size()) {
    if (_threads_busy[i]) {
      std::this_thread::sleep_for(_sleep_duration);
    } else {
      i++;
    }
  }
}
