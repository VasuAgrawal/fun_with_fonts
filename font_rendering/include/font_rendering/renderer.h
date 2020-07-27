#pragma once

#include <ft2build.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <tuple>

#include "freetype/config/ftoption.h"
#include FT_FREETYPE_H

#include <fmt/format.h>

#include <opencv2/core.hpp>

struct RenderStats {
 public:
  RenderStats() = default;

  bool font_not_loaded = false;
  std::vector<char> char_loads_failed;
  uint32_t out_of_image_bounds_count = 0;
  uint32_t out_of_cell_bounds_count = 0;
  uint32_t overwrites = 0;
  std::vector<std::pair<char, char>> matched_bitmaps;
  std::vector<char> empty_characters;

  static std::vector<std::string> makeMatchStrings(
      const std::vector<std::pair<char, char>>& matched_bitmaps);

  void update(const RenderStats& other);

  operator bool() const {
    // clang-format off
    return font_not_loaded || 
           !char_loads_failed.empty() ||
           out_of_image_bounds_count ||
           // out_of_cell_bounds_count ||
           overwrites ||
           !matched_bitmaps.empty() ||
           !empty_characters.empty();
    // clang-format on
  }
};

template <>
struct fmt::formatter<RenderStats> {
  constexpr auto parse(format_parse_context& ctx) {
    auto it = ctx.begin(), end = ctx.end();

    // Nothing to parse, so just move the iterator along.
    while (it != end && *it != '}') {
      ++it;
    }

    return it;
  }

  template <typename FormatContext>
  auto format(const RenderStats& stats, FormatContext& ctx) {
    format_to(ctx.out(), "RenderStats: [");
    if (stats.font_not_loaded) {
      format_to(ctx.out(), " font not loaded,");
    }

    if (!stats.char_loads_failed.empty()) {
      format_to(ctx.out(), " {} character loads failed ",
                stats.char_loads_failed.size());
      for (const auto c : stats.char_loads_failed) {
        format_to(ctx.out(), "'{}', ", c);
      }
    }

    if (stats.out_of_image_bounds_count) {
      format_to(ctx.out(), " {} pixels out of image bounds,",
                stats.out_of_image_bounds_count);
    }

    // if (stats.out_of_cell_bounds_count) {
    //   format_to(ctx.out(), " {} pixels out of cell bounds,",
    //   stats.out_of_cell_bounds_count);
    // }

    if (stats.overwrites) {
      format_to(ctx.out(), " {} pixels overwritten,", stats.overwrites);
    }

    if (!stats.matched_bitmaps.empty()) {
      format_to(ctx.out(), " {} matched bitmap pairs ",
                stats.matched_bitmaps.size());
      for (const auto [c1, c2] : stats.matched_bitmaps) {
        format_to(ctx.out(), "('{}', '{}'), ", c1, c2);
      }
    }

    if (!stats.empty_characters.empty()) {
      format_to(ctx.out(), " {} empty characters ",
                stats.empty_characters.size());
      for (const auto c : stats.empty_characters) {
        format_to(ctx.out(), "'{}', ", c);
      }
    }

    return format_to(ctx.out(), "]");
  }
};

enum class ImageWriteStyle : uint8_t {
  Fastest = 0,  // Uses .pgm to dump bits to disk as fast as possible
  Efficient,  // Uses LZ .tiff compression, gives most compression per unit time
  Smaller,    // Uses RLE .png compression, which is a little smaller than .tiff
  Smallest,   // Uses .webp compression, which is as small as I've gotten
  COUNT,      // Number of ImageWriteStyle
};

inline static const std::vector<std::string> DEFAULT_ATLAS{
    "ABCDEFGH", "IJKLMNOP", "QRSTUVWX", "YZabcdef",
    "ghijklmn", "opqrstuv", "wxyz0123", "456789?!",
};

// Atlas is composed of all of the printable ascii characters.
std::vector<std::string> makeFullAtlas();

class FtPtr {
 public:
  FtPtr() = default;
  FtPtr(FT_Library library) : ptr_(library) {}

  FtPtr(const FtPtr& other) = delete;  // Not a shard ptr
  FtPtr(FtPtr&& other) {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }

  FtPtr& operator=(const FtPtr& other) = delete;
  FtPtr& operator=(FtPtr&& other) {
    FT_Done_FreeType(ptr_);
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }

  ~FtPtr() { FT_Done_FreeType(ptr_); }

  operator FT_Library() const { return ptr_; }

 private:
  FT_Library ptr_ = nullptr;
};

struct RendererSpacing {
 public:
  static constexpr int DEFAULT = -1;  // Just some negative value
  int dpi;
  int point;

  // Compute these in the middle because we need them for atlas_border/padding
  // default values. User should never change the values, as it's determined
  // by the DPI / point settings.
  int em;
  int half_em;

  int atlas_border;
  int atlas_padding;

  explicit RendererSpacing(int dpi = DEFAULT, int point = DEFAULT,
                           int atlas_border = DEFAULT,
                           int atlas_padding = DEFAULT)
      : dpi(dpi > 0 ? dpi : 110),
        point(point > 0 ? point : 72),
        em(this->point * this->dpi / 72),
        half_em(em / 2),
        atlas_border(atlas_border >= 0 ? atlas_border : em / 8),
        atlas_padding(atlas_padding >= 0
                          ? atlas_padding
                          : std::max(em / 2, this->atlas_border)) {}
};

class Renderer {
  static constexpr size_t RELOAD_COUNT = 1000;

 public:
  explicit Renderer(RendererSpacing spacing = RendererSpacing());
  explicit Renderer(const std::vector<std::string>& user_atlas,
                    RendererSpacing spacing = RendererSpacing());
  ~Renderer();

  Renderer(const Renderer& other) = delete;
  Renderer(Renderer&& other);
  Renderer& operator=(const Renderer& other) = delete;
  Renderer& operator=(Renderer&& other);

  const std::vector<std::string>& getAtlas() const;
  std::string getAtlasString() const;

  std::tuple<cv::Mat, RenderStats> renderAtlas(bool cells = true,
                                               std::string_view highlight = "");
  bool loadFontFace(const std::string& path, int index = 0);

  static std::optional<std::filesystem::path> saveImage(
      const std::filesystem::path& dirname, const std::string& basename,
      const cv::Mat& mat, ImageWriteStyle style);

 private:
  RendererSpacing spacing_;
  std::vector<std::string> user_atlas_;
  size_t user_atlas_width_ = 0;

  size_t render_count_ = 0;
  FtPtr library_ = nullptr;
  FT_Face face_ = nullptr;
  std::string face_name_;
  int face_index_;
  bool loaded_ = false;
  int atlas_width_ = 0;
  int atlas_height_ = 0;
  size_t atlas_buffer_size_ = 0;
  std::unique_ptr<uint8_t[]> atlas_buffer_ = nullptr;
  std::vector<size_t> char_buffer_sizes_;
  std::vector<char> char_buffer_symbols_;
  std::vector<std::pair<int, int>> char_buffer_offsets_;
  std::vector<std::unique_ptr<uint8_t[]>> char_buffers_;

  void reloadFreeType();

  static RenderStats drawBitmap(
      cv::Mat& mat, FT_Bitmap& bitmap, int start_x, int start_y,
      int cell_left = std::numeric_limits<int>::min(),
      int cell_top = std::numeric_limits<int>::min(),
      int cell_right = std::numeric_limits<int>::max(),
      int cell_bot = std::numeric_limits<int>::max());
};
