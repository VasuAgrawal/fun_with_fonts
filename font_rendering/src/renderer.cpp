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
}

Renderer::Renderer(Renderer&& other) {
  *this = std::move(other);
}

Renderer& Renderer::operator=(Renderer&& other) {
  user_atlas_ = std::move(other.user_atlas_);
  user_atlas_width_ = other.user_atlas_width_;
  render_count_ = other.render_count_;

  library_ = std::move(other.library_);
  
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
  char_buffer_symbols_ = std::move(char_buffer_symbols_);

  return *this;
}

std::tuple<cv::Mat, RenderStats> Renderer::renderAtlas(bool cells, std::string_view highlight) {
  RenderStats stats;

  if (++render_count_ % RELOAD_COUNT == 0) {
    reloadFreeType();
    loadFontFace(face_name_, face_index_);
  }

  if (!loaded_) {
    stats.font_not_loaded = true;
    return std::make_tuple<cv::Mat, RenderStats>(
        {}, std::move(stats));
  }

  std::memset(atlas_buffer_.get(), 0, atlas_buffer_size_);
  cv::Mat mat(atlas_height_, atlas_width_, CV_8UC1, atlas_buffer_.get());
  char_buffers_.clear();
  char_buffer_sizes_.clear();
  char_buffer_symbols_.clear();
  char_buffer_offsets_.clear();

  // Continuous pen data, in case we don't want to draw each character in its
  // individual cell (which is the default).
  int cont_px = 0;
  int cont_py = 0;

  for (int row = 0; const auto& line : user_atlas_) {
    const auto cy = ATLAS_PADDING + row * (EM + ATLAS_BORDER) + HALF_EM;
    const auto cell_top = cy - HALF_EM - ATLAS_BORDER;
    const auto cell_bot = cy + HALF_EM + ATLAS_BORDER + 1;

    // Move the continuous pen to the middle of the bottom of the first cell
    // every time there's a new row.
    cont_py = cell_bot - ATLAS_BORDER;

    for (int col = 0; const auto& c : line) {
      if (FT_Load_Char(face_, c, FT_LOAD_RENDER)) {
        stats.char_loads_failed.push_back(c);
        continue;
      }

      auto slot = face_->glyph;
      auto& bitmap = slot->bitmap;
      
      const auto cx = ATLAS_PADDING + col * (EM + ATLAS_BORDER) + HALF_EM;

      // Move the pen to the start of the line for the new character.
      // Bottom left corner of the cell, effectively.
      if (col == 0) {
        cont_px = cx - HALF_EM;
      }

      // First, figure out where to place the character. This offset is computed
      // first so that we can use it to identify duplicates, as some characters
      // can have the same actual bitmap, but be rendered somewhere else.

      int offset_x = 0;
      int offset_y = 0;
      int px = 0;
      int py = 0;

      if (cells) {
        // Each character should get rendered to its own cell.
        
        // Offset from the center of the cell by the bitmap size. This generally
        // works, but makes things look "off" a bit when characters are supposed
        // to overlap (e.g. with a long swish from Q).
        // const auto px = cx - bitmap.width / 2;
        // const auto py = cy - bitmap.rows / 2;
        
        // Instead, try to correct the X position based on the pen advance
        // (which indicates where the next character should be start). We'll
        // also try to adjust Y so that characters look like they're sitting on
        // the line.
       
        offset_x = -((slot->advance.x >> 6) - slot->bitmap_left) / 2;
        offset_y = HALF_EM - slot->bitmap_top;

        px = cx + offset_x;
        py = cy + offset_y;

        // px = cx - ((slot->advance.x >> 6) - slot->bitmap_left) / 2;
        // py = cy + HALF_EM - slot->bitmap_top;
      } else {
        // Use the continuous pen, leaving no space between characters.
        offset_x = slot->bitmap_left;
        offset_y = -slot->bitmap_top;
      
        px = cont_px + offset_x;
        py = cont_py + offset_y;

        // px = cont_px + slot->bitmap_left;
        // py = cont_py - slot->bitmap_top;
      }




      // Skip the duplicate check for spaces.
      if (c != ' ') {
        // Check if the character is a duplicate by doing memory comparison
        const auto char_buffer_size = bitmap.rows * bitmap.width;
        char_buffer_sizes_.push_back(char_buffer_size);
        char_buffer_symbols_.push_back(c);
        const auto char_buffer_offset =
          char_buffer_offsets_.emplace_back(offset_x, offset_y);
        char_buffers_.push_back(std::make_unique<uint8_t[]>(char_buffer_size));
        auto& char_buffer = char_buffers_.back();
        std::memcpy(char_buffer.get(), bitmap.buffer, char_buffer_size);

        for (size_t i = 0; i < char_buffers_.size() - 1; ++i) {
          // They need to have the same size to be comparable. This is put first
          // since it's the most likely check to fail.
          auto cmp_size = char_buffer_sizes_[i];
          if (cmp_size != char_buffer_size) {
            continue;
          }

          // The offsets need to be the same. Otherwise, we have the same glyph
          // rendered at different locations, which is not considered duplicate.
          if (char_buffer_offsets_[i] != char_buffer_offset) {
            continue;
          }

          // The symbols need to be different. This isn't necessary if rendering
          // an atlas (as each symbols is rendered once), but is necessary if
          // rendering generic text.
          if (char_buffer_symbols_[i] == c) {
            continue;
          }

          // The bitmaps need to be identical.
          if (std::memcmp(char_buffers_[i].get(), char_buffer.get(),
                           cmp_size)) {
            continue;
          }
            // fmt::print("Matched buffer {} ({}) to buffer {} ({})\n",
            //            char_buffers_.size() - 1, c, i, char_buffer_symbols_[i]);
          stats.matched_bitmaps.emplace_back(char_buffer_symbols_[i], c);
          break;
        }
      }


      const auto cell_left = cx - HALF_EM - ATLAS_BORDER;
      const auto cell_right = cx + HALF_EM + ATLAS_BORDER + 1;

      // Determine if the character should get highlighted
      bool draw_highlight = false;
      for (const auto cmp : highlight) {
        if (c == cmp) {
          draw_highlight = true;
          break;
        }
      }

      if (draw_highlight) {
        if (cells) {
          cv::rectangle(mat, cv::Point(cell_left, cell_top), 
              cv::Point(cell_right, cell_bot), cv::Scalar(128), 2);
        }
      }

      // fmt::print("Character {} width {} left {} advance x {}, height {}  top {} advance y {}\n",
      //     c, bitmap.width, slot->bitmap_left, slot->advance.x >> 6, bitmap.rows, slot->bitmap_top, slot->advance.y >> 6);
      auto new_stats = drawBitmap(mat, bitmap, px, py,
                     cell_left, cell_top, cell_right, cell_bot);

      // Don't treat space as an empty character, but transfer over the rest of
      // the rendering stats.
      if (!new_stats.empty_characters.empty()) {
        if (c == ' ') {
          new_stats.empty_characters.clear();
        } else {
          new_stats.empty_characters[0] = c;
        }
      }

      stats.update(new_stats);

      cont_px += slot->advance.x >> 6;
      cont_py += slot->advance.y >> 6;

      ++col;
    }

    ++row;
  }

  return std::make_tuple<cv::Mat, RenderStats>(std::move(mat),
                                                 std::move(stats));
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
constexpr auto to_integral(E e) -> typename std::underlying_type_t<E> {
  return static_cast<typename std::underlying_type_t<E>>(e);
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

  FT_Library temp_library_;
  if (const auto error = FT_Init_FreeType(&temp_library_)) {
    throw std::runtime_error(
        fmt::format("Failed to initialize libfreetype2: {}", error));
  }
  library_ = temp_library_;
}

RenderStats Renderer::drawBitmap(cv::Mat& mat, FT_Bitmap& bitmap, int start_x,
                                   int start_y, int cell_left, int cell_top,
                                   int cell_right, int cell_bot) {
  RenderStats stats;
  uint64_t write_count = 0;

  // cv::circle(mat, cv::Point(start_x, start_y), 3, cv::Scalar(255), 1);
  // cv::rectangle(mat, cv::Point(cell_left, cell_top), cv::Point(cell_right, cell_bot),
  //     cv::Scalar(255), 2);

  for (int y = 0; y < bitmap.rows; ++y) {
    auto draw_y = start_y + y;
    stats.out_of_cell_bounds_count += (draw_y < cell_top) || (draw_y >= cell_bot);
    if (draw_y < 0 || draw_y >= mat.rows) {
      ++stats.out_of_image_bounds_count;
      continue;
    }

    for (int x = 0; x < bitmap.width; ++x) {
      auto draw_x = start_x + x;
      stats.out_of_cell_bounds_count += (draw_x < cell_left) || (draw_x >= cell_right);
      if (draw_x < 0 || draw_x >= mat.cols) {
        ++stats.out_of_image_bounds_count;
        continue;
      }

      if (bitmap.buffer[y * bitmap.width + x]) {
        // Add an overwrite count if there's a nonzero pixel at the location
        // we're going to write to.
        stats.overwrites += static_cast<bool>(mat.at<uint8_t>(draw_y, draw_x));

        // Alpha blending, courtesy of equation on wikipedia.
        // Note that I assume the color for the new 
        const float dst_rgb = mat.at<uint8_t>(draw_y, draw_x) / 255.0f;
        const float src_rgb = 255.0f; // white in mono
        const float src_a = bitmap.buffer[y * bitmap.width + x] / 255.0f;
        const uint8_t out_rgb = src_rgb * (1.0f * src_a + dst_rgb * (1.0f - src_a));
        mat.at<uint8_t>(draw_y, draw_x) = out_rgb;

        ++write_count;
      }
    }
  }

  if (!write_count) {
    // Abuse the container and dump something in here to indicate that the
    // caller should stuff the right character in.
    stats.empty_characters.push_back(0);
  }

  return stats;
}
