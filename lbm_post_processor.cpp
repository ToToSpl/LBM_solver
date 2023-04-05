#include "include/lbm_constants.h"
#include "include/lbm_types.h"
#include "png.h"
#include "zip.h"
#include <bits/stdc++.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
#include <vector>

#include "src/colors.h"

#define OUTPUT_DIR "./output"

#define OUTPUT_WIDTH 400
#define OUTPUT_HEIGHT 100
#define OUTPUT_DEPTH 1
#define OUTPUT_SLICE OUTPUT_DEPTH / 2

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

void create_rgb_image(const char *filename, u_int8_t **buffer, int width,
                      int height) {
  FILE *fp = fopen(filename, "wb");
  if (fp == NULL) {
    perror("Failed to open file for writing");
    return;
  }

  png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    fclose(fp);
    perror("Failed to create PNG write structure");
    return;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, NULL);
    fclose(fp);
    perror("Failed to create PNG info structure");
    return;
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    perror("Failed to set PNG error handler");
    return;
  }

  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);
  png_write_info(png_ptr, info_ptr);
  png_write_image(png_ptr, (png_bytep *)buffer);
  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
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

inline size_t index(size_t x, size_t y, size_t z) {
  return (z * OUTPUT_WIDTH * OUTPUT_HEIGHT) + (y * OUTPUT_WIDTH) + x;
}

int main() {

  auto outputs = files_in_output(OUTPUT_DIR);

  u_int8_t **rgb_buffer =
      (u_int8_t **)malloc(OUTPUT_HEIGHT * sizeof(u_int8_t *));
  for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
    rgb_buffer[y] = (u_int8_t *)malloc(OUTPUT_WIDTH * 3 * sizeof(u_int8_t));
    memset(rgb_buffer[y], 0, OUTPUT_WIDTH * 3 * sizeof(u_int8_t));
  }

  float cs = std::sqrt(1.f / 3.f);
  size_t z = OUTPUT_SLICE;
  for (size_t j = 0; j < outputs.size(); j++) {
    size_t i = j * 10;
    if (i % 20 != 0)
      continue;
    auto &p = outputs[j];
    if (p.find(".zip") != p.size() - 4)
      continue;

    LatticeOutputFile o = read_zip_file(p.c_str());
    std::cout << p << std::endl;
    for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
      for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
        auto &u = o.lattice[index(x, y, z)].u;
        float mag =
            std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z) / std::sqrt(CS2);
        float head = atan2(u.y, u.x);
        rgb color = hsv2rgb({head, 1.f, mag});

        rgb_buffer[y][3 * x + 0] = (u_int8_t)(255.f * color.r);
        rgb_buffer[y][3 * x + 1] = (u_int8_t)(255.f * color.g);
        rgb_buffer[y][3 * x + 2] = (u_int8_t)(255.f * color.b);
      }
    }
    std::string filename = "./mag/" + std::to_string(i) + ".png";
    create_rgb_image(filename.c_str(), rgb_buffer, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    free(o.lattice);
  }

  return 0;
}
