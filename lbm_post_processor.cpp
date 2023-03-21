#include "include/lbm_types.h"
#include <bits/stdc++.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "zip.h"

#define OUTPUT_DIR "./output"

typedef struct {
  LatticeOutput *lattice;
  size_t size;
} LatticeOutputFile;

LatticeOutputFile read_zip_file(const char *filename) {
  LatticeOutputFile result;
  int err;
  zip *zf = zip_open(filename, 0, &err);
  if (zf == nullptr) {
    std::cerr << "Failed to open zip file: " << zip_strerror(zf) << std::endl;
    return {nullptr, 0};
  }

  zip_int64_t idx = zip_name_locate(zf, filename, 0);
  if (idx == -1) {
    std::cerr << "Failed to locate zip entry: " << zip_strerror(zf)
              << std::endl;
    zip_close(zf);
    return {nullptr, 0};
  }

  struct zip_stat st;
  zip_stat_init(&st);
  zip_stat_index(zf, idx, 0, &st);

  result.lattice = (LatticeOutput *)malloc(st.size);

  zip_file *zf_entry = zip_fopen_index(zf, idx, 0);
  if (zf_entry == nullptr) {
    std::cerr << "Failed to open zip entry: " << zip_strerror(zf) << std::endl;
    free(result.lattice);
    zip_close(zf);
    return {nullptr, 0};
  }
  if (zip_fread(zf_entry, result.lattice, st.size) == -1) {
    std::cerr << "Failed to read zip entry: " << zip_file_strerror(zf_entry)
              << std::endl;
    free(result.lattice);
    zip_fclose(zf_entry);
    zip_close(zf);
    return {nullptr, 0};
  }
  zip_fclose(zf_entry);
  zip_close(zf);

  return result;
}

bool string_sort(std::string &a, std::string &b) {
  size_t num_a = std::stoll(a.substr(a.find("_") + 1, a.size()));
  size_t num_b = std::stoll(b.substr(b.find("_") + 1, b.size()));
  return num_a < num_b;
}

std::vector<std::string> files_in_output(std::string dir_path) {
  std::vector<std::string> outputs;
  for (const auto &entry : std::filesystem::directory_iterator(dir_path))
    outputs.push_back(entry.path());

  std::sort(outputs.begin(), outputs.end(), string_sort);
  return outputs;
}

int main() {

  auto outputs = files_in_output(OUTPUT_DIR);

  for (const auto &p : outputs) {
    LatticeOutputFile output = read_zip_file(p.c_str());
    std::cout << p << "\t" << output.lattice[0].rho << std::endl;
    free(output.lattice);
  }

  return 0;
}
