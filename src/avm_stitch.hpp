#pragma once

#include "avm_asset.hpp"

#include <cudaEGL.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

enum class AvmFitMode : int { Contain = 0, Cover = 1, Stretch = 2 };

class AvmStitcher {
 public:
  AvmStitcher() = default;
  ~AvmStitcher();
  AvmStitcher(const AvmStitcher &) = delete;
  AvmStitcher &operator=(const AvmStitcher &) = delete;

  bool initialize(const AvmAsset &asset, uint32_t output_width, uint32_t output_height,
                  AvmFitMode fit_mode, std::string &error);
  void reset();
  bool process(CUeglFrame inputs[4], CUeglFrame &output, std::string &error);

 private:
  uint32_t source_width_ = 0, source_height_ = 0;
  uint32_t canvas_width_ = 0, canvas_height_ = 0;
  uint32_t output_width_ = 0, output_height_ = 0;
  AvmFitMode fit_mode_ = AvmFitMode::Contain;
  float *map_x_ = nullptr, *map_y_ = nullptr, *weights_ = nullptr;
  uint8_t *overlay_ = nullptr;
  uint8_t *input_y_[4]{}, *input_uv_[4]{};
  size_t input_y_pitch_[4]{}, input_uv_pitch_[4]{};
  uint8_t *output_y_ = nullptr, *output_uv_ = nullptr;
  size_t output_y_pitch_ = 0, output_uv_pitch_ = 0;
  cudaStream_t stream_ = nullptr;
};
