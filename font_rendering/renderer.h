#include <fmt/format.h>
#include <ft2build.h>

#include <cstring>
#include <memory>
#include <optional>
#include FT_FREETYPE_H
#include <opencv2/core.hpp>

class Renderer {
  static constexpr int DPI = 110;
  static constexpr int POINT = 72;
  static constexpr int EM = POINT * DPI / 72;
  static constexpr int HALF_EM = EM / 2;
  static constexpr int ATLAS_BORDER = EM / 8;  // Border for each cell
  static constexpr int ATLAS_PADDING = std::max(EM / 2, ATLAS_BORDER);  // side

  // static constexpr int ATLAS_WIDTH = 8;
  // inline static const std::vector<std::string> ATLAS{
  //     "ABCDEFGH", "IJKLMNOP", "QRSTUVWX", "YZabcdef",
  //     "ghijklmn", "opqrstuv", "wxyz0123", "456789?!",
  // };

  static constexpr int ATLAS_WIDTH = 6;
  inline static const std::vector<std::string> ATLAS{
      "ABCDEF", "GHIJKL", "MNOPQR", "STUVWX", "YZ0123", "456789",
  };

 public:
  Renderer() {
    if (const auto error = FT_Init_FreeType(&library_)) {
      throw std::runtime_error(
          fmt::format("Failed to initialize libfreetype2: {}", error));
    }

    auto dim = [this](int border, int padding, int count) {
      return 2 * padding + std::max(count - 1, 0) * border + count * EM;
    };
    atlas_width_ = dim(ATLAS_BORDER, ATLAS_PADDING, ATLAS_WIDTH);
    atlas_height_ = dim(ATLAS_BORDER, ATLAS_PADDING, ATLAS.size());
    atlas_buffer_size_ = atlas_width_ * atlas_height_;
    atlas_buffer_ = std::make_unique<uint8_t[]>(atlas_buffer_size_);
  }

  ~Renderer() {
    FT_Done_Face(face_);
    FT_Done_FreeType(library_);
  }

  Renderer(const Renderer& other) = delete;
  Renderer(Renderer&& other) = default;
  Renderer& operator=(const Renderer& other) = delete;
  Renderer& operator=(Renderer&& other) = default;

  std::optional<cv::Mat> renderAtlas() {
    if (!loaded_) {
      return std::nullopt;
    }

    std::memset(atlas_buffer_.get(), 0, atlas_buffer_size_);
    cv::Mat mat(atlas_height_, atlas_width_, CV_8UC1, atlas_buffer_.get());
    // cv::Mat mat = cv::Mat::zeros(atlas_height_, atlas_width_, CV_8UC1);

    for (int row = 0; const auto& line : ATLAS) {
      const auto cy = ATLAS_PADDING + row * (EM + ATLAS_BORDER) + HALF_EM;

      for (int col = 0; const auto& c : line) {
        if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
          return std::nullopt;
        }

        auto slot = face_->glyph;
        auto& bitmap = slot->bitmap;
        const auto cx = ATLAS_PADDING + col * (EM + ATLAS_BORDER) + HALF_EM;
        drawBitmap(mat, bitmap, cx - bitmap.width / 2, cy - bitmap.rows / 2);

        ++col;
      }

      ++row;
    }

    return mat;
  }

  std::optional<cv::Mat> renderTestText() {
    if (!loaded_) {
      return std::nullopt;
    }

    // 50 pt at 100 dpi
    if (FT_Set_Char_Size(face_, POINT * 64, 0, DPI, 0)) {
      return std::nullopt;
    }

    cv::Mat mat = cv::Mat::zeros(1000, 2000, CV_8UC1);
    std::string text = "The quick brown fox jumps over the lazy dog!";

    int px = 100;
    int py = 200;

    for (const char c : text) {
      if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
        return std::nullopt;
      }

      auto slot = face_->glyph;
      auto& bitmap = slot->bitmap;

      drawBitmap(mat, slot->bitmap, px + slot->bitmap_left,
                 py - slot->bitmap_top);
      drawBitmap(mat, slot->bitmap, px + slot->bitmap_left,
                 100 + py - slot->bitmap_top);
      drawBitmap(mat, slot->bitmap, px + slot->bitmap_left,
                 200 + py - slot->bitmap_top);
      drawBitmap(mat, slot->bitmap, px + slot->bitmap_left,
                 300 + py - slot->bitmap_top);
      drawBitmap(mat, slot->bitmap, px + slot->bitmap_left,
                 400 + py - slot->bitmap_top);
      drawBitmap(mat, slot->bitmap, px + slot->bitmap_left,
                 500 + py - slot->bitmap_top);

      px += slot->advance.x >> 6;
      py += slot->advance.y >> 6;
    }

    return mat;
  }

  bool loadFontFace(const std::string& path, int index = 0) {
    loaded_ = true;
    loaded_ &= !FT_New_Face(library_, path.c_str(), index, &face_);
    loaded_ &= !FT_Set_Char_Size(face_, POINT * 64, 0, DPI, 0);
    return loaded_;
  }

  const auto& getLibrary() { return library_; }

  const auto& getFontFace() { return face_; }

 private:
  FT_Library library_ = nullptr;
  FT_Face face_ = nullptr;
  bool loaded_ = false;
  int atlas_width_ = 0;
  int atlas_height_ = 0;
  size_t atlas_buffer_size_ = 0;
  std::unique_ptr<uint8_t[]> atlas_buffer_ = nullptr;

  static void drawBitmap(cv::Mat& mat, FT_Bitmap& bitmap, int start_x,
                         int start_y) {
    for (int y = 0; y < bitmap.rows; ++y) {
      auto draw_y = start_y + y;
      if (draw_y < 0 || draw_y >= mat.rows) {
        continue;
      }

      for (int x = 0; x < bitmap.width; ++x) {
        auto draw_x = start_x + x;
        if (draw_x < 0 || draw_x >= mat.cols) {
          continue;
        }

        mat.at<uint8_t>(draw_y, draw_x) |= bitmap.buffer[y * bitmap.width + x];
      }
    }
  }
};
