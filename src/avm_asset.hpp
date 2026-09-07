#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Precomputed AVM v1 asset. All per-frame geometry is reduced to four inverse
// source maps plus normalized blend weights. Overlay pixels are stored as
// full-resolution limited-range YUVA so the CUDA runtime never needs BGR.
struct AvmAssetHeader {
  char magic[8];              // "AVMAP01\0"
  uint32_t version;           // 1
  uint32_t source_width;      // 1920
  uint32_t source_height;     // 1080
  uint32_t canvas_width;      // 1216
  uint32_t canvas_height;     // 1436
  uint32_t camera_count;      // 4: front,left,right,bottom
  uint32_t flags;             // bit 0: overlay present
  uint64_t pixel_count;
  uint64_t reserved[4];
};

static_assert(sizeof(AvmAssetHeader) == 80, "AVM header ABI changed");

struct AvmAsset {
  AvmAssetHeader header{};
  // Camera-major arrays. Index = camera * pixel_count + output_pixel.
  std::vector<float> map_x;
  std::vector<float> map_y;
  std::vector<float> weights;
  // Pixel-major limited-range Y, U, V and alpha.
  std::vector<uint8_t> overlay_yuva;
};

bool load_avm_asset(const std::string &path, AvmAsset &asset, std::string &error);
