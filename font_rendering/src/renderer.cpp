#include "font_rendering/renderer.h"

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <type_traits>

Renderer::Renderer() : Renderer(DEFAULT_ATLAS) {};

Renderer::Renderer(const std::vector<std::string>& user_atlas) : user_atlas_(user_atlas) {
  for (const auto& line : user_atlas_) {
    user_atlas_width_ = std::max(user_atlas_width_, line.size());
    fmt::print("{}\n", line);
  }


  reloadFreeType();

  auto dim = [this](int border, int padding, int count) {
    return 2 * padding + std::max(count - 1, 0) * border + count * EM;
  };
  atlas_width_ = dim(ATLAS_BORDER, ATLAS_PADDING, user_atlas_width_);
  atlas_height_ = dim(ATLAS_BORDER, ATLAS_PADDING, user_atlas_.size());
  atlas_buffer_size_ = atlas_width_ * atlas_height_;
  atlas_buffer_ = std::make_unique<uint8_t[]>(atlas_buffer_size_);
}

Renderer::~Renderer() {
  FT_Done_Face(face_);
  FT_Done_FreeType(library_);
}

Renderer& Renderer::operator=(Renderer&& other) {
  user_atlas_ = std::move(other.user_atlas_);
  user_atlas_width_ = other.user_atlas_width_;
  render_count_ = other.render_count_;

  FT_Done_FreeType(library_);
  library_ = other.library_;
  other.library_ = nullptr;
  
  FT_Done_Face(face_);
  face_ = other.face_;
  other.face_ = nullptr;

  face_name_ = std::move(other.face_name_);
  face_index_ = other.face_index_;
  loaded_ = other.loaded_;
  atlas_width_ = other.atlas_width_;
  atlas_height_ = other.atlas_height_;
  atlas_buffer_size_ = other.atlas_buffer_size_;
  atlas_buffer_ = std::move(other.atlas_buffer_);
  char_buffers_ = std::move(other.char_buffers_);
  char_buffer_sizes_ = std::move(char_buffer_sizes_);

  return *this;
}

std::tuple<cv::Mat, RendererError> Renderer::renderAtlas() {
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

  for (int row = 0; const auto& line : user_atlas_) {
    const auto cy = ATLAS_PADDING + row * (EM + ATLAS_BORDER) + HALF_EM;
    const auto cell_top = cy - HALF_EM - ATLAS_BORDER;
    const auto cell_bot = cy + HALF_EM + ATLAS_BORDER + 1;

    for (int col = 0; const auto& c : line) {
      if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
        first_error = first_error == RendererError::None
                          ? RendererError::CharLoadFailed
                          : first_error;
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
        char_buffers_.push_back(std::make_unique<uint8_t[]>(char_buffer_size));
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

      // const auto px = cx - bitmap.width / 2;
      // const auto py = cy - bitmap.rows / 2;

      const auto px = cx - ((slot->advance.x >> 6) - slot->bitmap_left) / 2;
      // const auto px = cx - bitmap.width / 2;
      const auto py = cy + HALF_EM - slot->bitmap_top;
      // fmt::print("Character {} width {} left {} advance x {}, height {}  top {} advance y {}\n",
      //     c, bitmap.width, slot->bitmap_left, slot->advance.x >> 6, bitmap.rows, slot->bitmap_top, slot->advance.y >> 6);
      auto e =
          drawBitmap(mat, bitmap, px, py,
                     cell_left, cell_top, cell_right, cell_bot);

      first_error = first_error == RendererError::None ? e : first_error;

      ++col;
    }

    ++row;
  }

  return std::make_tuple<cv::Mat, RendererError>(std::move(mat),
                                                 std::move(first_error));
}

bool Renderer::loadFontFace(const std::string& path, int index) {
  loaded_ = true;
  loaded_ &= !FT_New_Face(library_, path.c_str(), index, &face_);
  loaded_ &= !FT_Set_Char_Size(face_, POINT * 64, 0, DPI, 0);

  if (loaded_) {
    face_name_ = path;
    face_index_ = index;
  }

  return loaded_;
}

template <typename E>
constexpr auto to_integral(E e) -> typename std::underlying_type<E>::type {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

std::optional<std::filesystem::path> Renderer::saveImage(
    const std::filesystem::path& dirname, const std::string& basename,
    const cv::Mat& mat, ImageWriteStyle style) {
  if (mat.empty()) {
    return std::nullopt;
  }

  static const std::vector<int> PARAMS[to_integral(ImageWriteStyle::COUNT)] = {
      [to_integral(ImageWriteStyle::Fastest)] =
          std::vector<int>{cv::IMWRITE_PXM_BINARY, 1},
      [to_integral(ImageWriteStyle::Efficient)] =
          std::vector<int>{cv::IMWRITE_TIFF_COMPRESSION, 5},
      [to_integral(ImageWriteStyle::Smaller)] =
          std::vector<int>{
              cv::IMWRITE_PNG_COMPRESSION,
              3,
              cv::IMWRITE_PNG_STRATEGY,
              3,
          },
      [to_integral(ImageWriteStyle::Smallest)] =
          std::vector<int>{
              cv::IMWRITE_WEBP_QUALITY,
              100,
          },
  };

  static const std::string EXTENSIONS[to_integral(ImageWriteStyle::COUNT)] = {
      ".pgm",
      ".tiff",
      ".png",
      ".webp",
  };

  const auto filename = dirname / (basename + EXTENSIONS[to_integral(style)]);
  const auto compression_params = PARAMS[to_integral(style)];

  if (cv::imwrite(filename.string(), mat, compression_params)) {
    return filename;
  }

  return std::nullopt;
}

void Renderer::reloadFreeType() {
  FT_Done_Face(face_);
  FT_Done_FreeType(library_);

  if (const auto error = FT_Init_FreeType(&library_)) {
    throw std::runtime_error(
        fmt::format("Failed to initialize libfreetype2: {}", error));
  }
}

RendererError Renderer::drawBitmap(cv::Mat& mat, FT_Bitmap& bitmap, int start_x,
                                   int start_y, int cell_left, int cell_top,
                                   int cell_right, int cell_bot) {
  RendererError e = RendererError::None;
  uint64_t write_count = 0;

  bool draw_circle = false;
  cv::Point circle;

  cv::rectangle(mat, cv::Point(cell_left, cell_top), cv::Point(cell_right, cell_bot),
      cv::Scalar(255), 2);

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
        if (mat.at<uint8_t>(draw_y, draw_x)) {    // and there's something there

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
