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

#define OUTPUT_WIDTH 6000
#define OUTPUT_HEIGHT 4000
#define OUTPUT_DEPTH 1
#define OUTPUT_SLICE (OUTPUT_DEPTH / 2)

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
  size_t num_a = std::stoll(a.substr(a.find_last_of("_") + 1, a.size()));
  size_t num_b = std::stoll(b.substr(b.find_last_of("_") + 1, b.size()));
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

void save_images(std::vector<std::string> &outputs) {
  u_int8_t **rgb_buffer =
      (u_int8_t **)malloc(OUTPUT_HEIGHT * sizeof(u_int8_t *));
  for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
    rgb_buffer[y] = (u_int8_t *)malloc(OUTPUT_WIDTH * 3 * sizeof(u_int8_t));
    memset(rgb_buffer[y], 0, OUTPUT_WIDTH * 3 * sizeof(u_int8_t));
  }

  float cs = std::sqrt(1.f / 3.f);
  size_t z = OUTPUT_SLICE;
  for (size_t j = 0; j < outputs.size(); j++) {
    size_t i = j * 100;
    // if (i % 20 != 0)
    //   continue;
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
        // float head = atan2(u.y, u.x);
        // rgb color = hsv2rgb({head, 1.f, mag});
        rgb color = {mag, mag, mag};

        rgb_buffer[y][3 * x + 0] = (u_int8_t)(255.f * color.r);
        rgb_buffer[y][3 * x + 1] = (u_int8_t)(255.f * color.g);
        rgb_buffer[y][3 * x + 2] = (u_int8_t)(255.f * color.b);
      }
    }
    std::string filename = "./mag/" + std::to_string(i) + ".png";
    create_rgb_image(filename.c_str(), rgb_buffer, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    free(o.lattice);
  }
}

void print_coefs(std::vector<std::string> &outputs) {

  std::cout << "NAME,CL,CD,CL/CD" << std::endl;

  float cs = std::sqrt(1.f / 3.f);
  size_t z = OUTPUT_SLICE;
  for (size_t j = 0; j < outputs.size(); j++) {
    size_t i = j * 100;
    // if (i % 20 != 0)
    //   continue;
    auto &p = outputs[j];
    if (p.find(".zip") != p.size() - 4)
      continue;

    LatticeOutputFile o = read_zip_file(p.c_str());
    std::cout << p << ",";
    float cd = 0.0f;
    float cl = 0.0f;
    for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
      for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
        auto &u = o.lattice[index(x, y, z)].u;
        cl += o.lattice[index(x, y, z)].m.y;
        cd += o.lattice[index(x, y, z)].m.x;
      }
    }
    cl = fabs(cl);
    cd = fabs(cd);
    std::cout << cl << "," << cd << "," << cl / cd << std::endl;
    free(o.lattice);
  }
}

float reverse_float(const float inFloat) {
  float retVal;
  char *floatToConvert = (char *)&inFloat;
  char *returnFloat = (char *)&retVal;

  // swap the bytes into a temporary buffer
  returnFloat[0] = floatToConvert[3];
  returnFloat[1] = floatToConvert[2];
  returnFloat[2] = floatToConvert[1];
  returnFloat[3] = floatToConvert[0];

  return retVal;
}

void save_as_vtk(std::string input, std::string output) {
  const char *header1 = "# vtk DataFile Version 2.0\n"
                        "speeds and density in sample\n"
                        "BINARY\n"
                        "DATASET STRUCTURED_POINTS\n"
                        "DIMENSIONS 6000 4000 1\n"
                        "ORIGIN 0 0 0\n"
                        "SPACING 1 1 1\n"
                        "POINT_DATA 24000000\n"
                        "VECTORS velocities float\n";
  const char *header2 = "SCALARS densities float 1\n"
                        "LOOKUP_TABLE default\n";
#ifdef LBM_MOMENT_EXCHANGE
  const char *header3 = "VECTORS momentum float\n";
#endif
  LatticeOutputFile o = read_zip_file(input.c_str());
  if (o.lattice == nullptr)
    return;

  std::ofstream file(output.c_str(), std::ios::binary);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(header1),
               sizeof(char) * strlen(header1));
    for (size_t z = 0; z < OUTPUT_DEPTH; z++) {
      for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
        for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
          auto &u = o.lattice[index(x, y, z)].u;
          float t = reverse_float(u.x);
          file.write(reinterpret_cast<const char *>(&t), sizeof(float));
          t = reverse_float(u.y);
          file.write(reinterpret_cast<const char *>(&t), sizeof(float));
          t = reverse_float(u.z);
          file.write(reinterpret_cast<const char *>(&t), sizeof(float));
        }
      }
    }

    file.write(reinterpret_cast<const char *>(header2),
               sizeof(char) * strlen(header2));
    for (size_t z = 0; z < OUTPUT_DEPTH; z++) {
      for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
        for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
          auto &rho = o.lattice[index(x, y, z)].rho;
          float t = reverse_float(rho);
          file.write(reinterpret_cast<const char *>(&t), sizeof(rho));
        }
      }
    }

#ifdef LBM_MOMENT_EXCHANGE
    file.write(reinterpret_cast<const char *>(header3),
               sizeof(char) * strlen(header3));
    for (size_t z = 0; z < OUTPUT_DEPTH; z++) {
      for (size_t y = 0; y < OUTPUT_HEIGHT; y++) {
        for (size_t x = 0; x < OUTPUT_WIDTH; x++) {
          auto &m = o.lattice[index(x, y, z)].m;
          float t = reverse_float(m.x);
          file.write(reinterpret_cast<const char *>(&t), sizeof(float));
          t = reverse_float(m.y);
          file.write(reinterpret_cast<const char *>(&t), sizeof(float));
          t = reverse_float(m.z);
          file.write(reinterpret_cast<const char *>(&t), sizeof(float));
        }
      }
    }
#endif

    file.close();
    std::cout << "Binary data written successfully." << std::endl;
  } else {
    std::cerr << "Unable to open the file." << std::endl;
  }

  free(o.lattice);
}

int main() {

  auto outputs = files_in_output(OUTPUT_DIR);
  // save_images(outputs);
  print_coefs(outputs);

  // save_as_vtk("./output/sample_300000.zip", "./out.vtk");

  return 0;
}
