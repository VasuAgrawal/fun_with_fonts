#include <fmt/format.h>
#include <ft2build.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include FT_FREETYPE_H
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
inline static const std::vector<std::string> RendererErrorNames {
  "None",
  "CharLoadFailed",
  "OutOfImageBounds",
  "OutOfCellBounds",
  "Overwrite",
  "Duplicate",
  "FontNotLoaded",
  "EmptyCharacter",
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
  Renderer() {
    reloadFreeType();

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

  std::tuple<cv::Mat, RendererError> renderAtlas() {
    if (++render_count_ % RELOAD_COUNT == 0) {
      reloadFreeType();
      loadFontFace(face_name_, face_index_);
    }

    if (!loaded_) {
      return std::make_tuple<cv::Mat, RendererError>(
          {}, RendererError::FontNotLoaded);
    }

    std::memset(atlas_buffer_.get(), 0, atlas_buffer_size_);
    cv::Mat mat(atlas_height_, atlas_width_, CV_8UC1, atlas_buffer_.get());
    char_buffers_.clear();
    char_buffer_sizes_.clear();

    RendererError first_error = RendererError::None;
    bool matched_buffer = false;

    for (int row = 0; const auto& line : ATLAS) {
      const auto cy = ATLAS_PADDING + row * (EM + ATLAS_BORDER) + HALF_EM;
      const auto cell_top = cy - HALF_EM - ATLAS_BORDER;
      const auto cell_bot = cy + HALF_EM + ATLAS_BORDER + 1;

      for (int col = 0; const auto& c : line) {
        if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
          first_error = first_error == RendererError::None ? RendererError::CharLoadFailed : first_error;
          continue;
          // return std::make_tuple<cv::Mat, RendererError>(
          //     {}, RendererError::CharLoadFailed);
        }

        auto slot = face_->glyph;
        auto& bitmap = slot->bitmap;

        if (!matched_buffer) {
          // Check if the character is a duplicate by doing memory comparison
          const auto char_buffer_size = bitmap.rows * bitmap.width;
          char_buffer_sizes_.push_back(char_buffer_size);
          char_buffers_.push_back(
              std::make_unique<uint8_t[]>(char_buffer_size));
          auto& char_buffer = char_buffers_.back();
          std::memcpy(char_buffer.get(), bitmap.buffer, char_buffer_size);

          for (size_t i = 0; i < char_buffers_.size() - 1; ++i) {
            // they need to have the same size to be comparable
            auto cmp_size = char_buffer_sizes_[i];
            if (cmp_size != char_buffer_size) {
              continue;
            }

            if (!std::memcmp(char_buffers_[i].get(), char_buffer.get(),
                             cmp_size)) {
              matched_buffer = true;
              // fmt::print("Matched buffer {} ({}) to buffer {} ({})\n",
              //            char_buffers_.size() - 1, c, i,
              //            ATLAS[i / ATLAS_WIDTH][i % ATLAS_WIDTH]);
              first_error = first_error == RendererError::None
                                ? RendererError::Duplicate
                                : first_error;
              break;
            }
          }
        }

        const auto cx = ATLAS_PADDING + col * (EM + ATLAS_BORDER) + HALF_EM;
        const auto cell_left = cx - HALF_EM - ATLAS_BORDER;
        const auto cell_right = cx + HALF_EM + ATLAS_BORDER + 1;
        auto e =
            drawBitmap(mat, bitmap, cx - bitmap.width / 2, cy - bitmap.rows / 2,
                       cell_left, cell_top, cell_right, cell_bot);

        first_error = first_error == RendererError::None ? e : first_error;

        ++col;
      }

      ++row;
    }

    return std::make_tuple<cv::Mat, RendererError>(std::move(mat),
                                                   std::move(first_error));
  }

  bool loadFontFace(const std::string& path, int index = 0) {
    loaded_ = true;
    loaded_ &= !FT_New_Face(library_, path.c_str(), index, &face_);
    loaded_ &= !FT_Set_Char_Size(face_, POINT * 64, 0, DPI, 0);

    if (loaded_) {
      face_name_ = path;
      face_index_ = index;
    }

    return loaded_;
  }

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

  void reloadFreeType() {
    FT_Done_Face(face_);
    FT_Done_FreeType(library_);

    if (const auto error = FT_Init_FreeType(&library_)) {
      throw std::runtime_error(
          fmt::format("Failed to initialize libfreetype2: {}", error));
    }
  }

  static RendererError drawBitmap(
      cv::Mat& mat, FT_Bitmap& bitmap, int start_x, int start_y,
      int cell_left = std::numeric_limits<int>::min(),
      int cell_top = std::numeric_limits<int>::min(),
      int cell_right = std::numeric_limits<int>::max(),
      int cell_bot = std::numeric_limits<int>::max()) {
    RendererError e = RendererError::None;
    uint64_t write_count = 0;

    bool draw_circle = false;
    cv::Point circle;

    for (int y = 0; y < bitmap.rows; ++y) {
      auto draw_y = start_y + y;
      if (draw_y < 0 || draw_y >= mat.rows) {
        e = e == RendererError::None ? RendererError::OutOfImageBounds : e;
        continue;
        // } else if (draw_y < cell_top || draw_y >= cell_bot) {
        //   e = e == RendererError::None ? RendererError::OutOfCellBounds : e;
      }

      for (int x = 0; x < bitmap.width; ++x) {
        auto draw_x = start_x + x;
        if (draw_x < 0 || draw_x >= mat.cols) {
          e = e == RendererError::None ? RendererError::OutOfImageBounds : e;
          continue;
          // } else if (draw_x < cell_left || draw_x >= cell_right) {
          //   e = e == RendererError::None ? RendererError::OutOfCellBounds :
          //   e;
        }

        if (bitmap.buffer[y * bitmap.width + x]) {  // If we need to draw
          if (mat.at<uint8_t>(draw_y, draw_x)) {  // and there's something there

            if (!draw_circle) {
              circle = cv::Point(draw_x, draw_y);
              draw_circle = true;
            }

            e = e == RendererError::None ? RendererError::Overwrite : e;
            // fmt::print("Overwriting at ({}, {}), old: {}, new: {}\n", draw_x,
            // draw_y,
            //     mat.at<uint8_t>(draw_y, draw_x), bitmap.buffer[y *
            //     bitmap.width + x]
            //     );
          }

          // This doesn't properly implement alpha blending, but since we flag
          // when there's an overwite, there's no need to. Overwriting the
          // default (black) value with this value is correctly blending.
          mat.at<uint8_t>(draw_y, draw_x) = bitmap.buffer[y * bitmap.width + x];
          ++write_count;
        }
      }
    }

    if (draw_circle) {
      // cv::circle(mat, circle, 50, cv::Scalar(1), 15);
      // cv::circle(mat, circle, 50, cv::Scalar(255), 5);
    }

    if (!write_count) {
      e = e == RendererError::None ? RendererError::EmptyCharacter : e;
    }

    return e;
  }
};
