#include "avm_asset.hpp"

#include <cstring>
#include <fstream>
#include <limits>

namespace {
template <typename T>
bool read_vector(std::ifstream &in, std::vector<T> &out, size_t count) {
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
  out.resize(count);
  in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
  return in.good();
}
}  // namespace

bool load_avm_asset(const std::string &path, AvmAsset &asset, std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) { error = "cannot open AVM asset: " + path; return false; }
  AvmAsset next{};
  in.read(reinterpret_cast<char *>(&next.header), sizeof(next.header));
  if (!in.good()) { error = "short AVM header"; return false; }
  if (std::memcmp(next.header.magic, "AVMAP01", 7) != 0 || next.header.version != 1) {
    error = "unsupported AVM asset magic/version"; return false;
  }
  if (next.header.camera_count != 4 || !next.header.source_width || !next.header.source_height ||
      !next.header.canvas_width || !next.header.canvas_height) {
    error = "invalid AVM dimensions/camera count"; return false;
  }
  const uint64_t expected = static_cast<uint64_t>(next.header.canvas_width) * next.header.canvas_height;
  if (next.header.pixel_count != expected || expected > 100000000ULL) {
    error = "invalid AVM pixel count"; return false;
  }
  const size_t pixels = static_cast<size_t>(expected);
  const size_t camera_pixels = pixels * 4U;
  if (!read_vector(in, next.map_x, camera_pixels) ||
      !read_vector(in, next.map_y, camera_pixels) ||
      !read_vector(in, next.weights, camera_pixels)) {
    error = "truncated AVM map/weight payload"; return false;
  }
  if (next.header.flags & 1U) {
    if (!read_vector(in, next.overlay_yuva, pixels * 4U)) {
      error = "truncated AVM overlay payload"; return false;
    }
  } else {
    next.overlay_yuva.assign(pixels * 4U, 0U);
  }
  char extra = 0;
  if (in.read(&extra, 1)) { error = "unexpected trailing bytes in AVM asset"; return false; }
  asset = std::move(next);
  return true;
}
