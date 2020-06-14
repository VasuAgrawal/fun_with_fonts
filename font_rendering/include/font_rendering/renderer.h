#pragma once

#include <ft2build.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include FT_FREETYPE_H

#include <opencv2/core.hpp>

enum class RendererError : size_t {
  None = 0,
  CharLoadFailed,
  OutOfImageBounds,
  OutOfCellBounds,
  Overwrite,
  Duplicate,
  FontNotLoaded,
  EmptyCharacter,
};
inline static const std::vector<std::string> RendererErrorNames{
    "None",      "CharLoadFailed", "OutOfImageBounds", "OutOfCellBounds",
    "Overwrite", "Duplicate",      "FontNotLoaded",    "EmptyCharacter",
};
inline static const std::vector<std::string> RendererErrorStrings{
    "none",
    "could not load character from font face",
    "tried to draw a character outside of image bounds",
    "tried to draw a character outside of cell bounds",
    "tried to overwrite existing drawn data",
    "found a character that was an exact duplicate of another",
    "font face isn't loaded",
    "nothing rendered when drawing a character",
};

enum class ImageWriteStyle : uint8_t {
  Fastest = 0,  // Uses .pgm to dump bits to disk as fast as possible
  Efficient,  // Uses LZ .tiff compression, gives most compression per unit time
  Smaller,    // Uses RLE .png compression, which is a little smaller than .tiff
  Smallest,   // Uses .webp compression, which is as small as I've gotten
  COUNT,      // Number of ImageWriteStyle
};

class Renderer {
  static constexpr size_t RELOAD_COUNT = 1000;

  static constexpr int DPI = 110;
  static constexpr int POINT = 72;
  static constexpr int EM = POINT * DPI / 72;
  static constexpr int HALF_EM = EM / 2;
  static constexpr int ATLAS_BORDER = EM / 8;  // Border for each cell
  static constexpr int ATLAS_PADDING = std::max(EM / 2, ATLAS_BORDER);  // side

  static constexpr int ATLAS_WIDTH = 8;
  inline static const std::vector<std::string> ATLAS{
      "ABCDEFGH", "IJKLMNOP", "QRSTUVWX", "YZabcdef",
      "ghijklmn", "opqrstuv", "wxyz0123", "456789?!",
  };

  // static constexpr int ATLAS_WIDTH = 6;
  // inline static const std::vector<std::string> ATLAS{
  //     "ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ0123", "456789",
  // };

 public:
  Renderer();
  ~Renderer();

  Renderer(const Renderer& other) = delete;
  Renderer(Renderer&& other) = default;
  Renderer& operator=(const Renderer& other) = delete;
  Renderer& operator=(Renderer&& other) = default;

  std::tuple<cv::Mat, RendererError> renderAtlas();
  bool loadFontFace(const std::string& path, int index = 0);
  
  static std::optional<std::filesystem::path> saveImage(
      const std::filesystem::path& dirname, const std::string& basename,
      const cv::Mat& mat, ImageWriteStyle style);

 private:
  size_t render_count_ = 0;
  FT_Library library_ = nullptr;
  FT_Face face_ = nullptr;
  std::string face_name_;
  int face_index_;
  bool loaded_ = false;
  int atlas_width_ = 0;
  int atlas_height_ = 0;
  size_t atlas_buffer_size_ = 0;
  std::unique_ptr<uint8_t[]> atlas_buffer_ = nullptr;
  std::vector<std::unique_ptr<uint8_t[]>> char_buffers_;
  std::vector<size_t> char_buffer_sizes_;

  void reloadFreeType();

  static RendererError drawBitmap(
      cv::Mat& mat, FT_Bitmap& bitmap, int start_x, int start_y,
      int cell_left = std::numeric_limits<int>::min(),
      int cell_top = std::numeric_limits<int>::min(),
      int cell_right = std::numeric_limits<int>::max(),
      int cell_bot = std::numeric_limits<int>::max());
};
