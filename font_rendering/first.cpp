#include <ft2build.h>
#include FT_FREETYPE_H

#include <fmt/format.h>
#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_path, "", "Path to font");

#define CHECK_FT(expr, err)                               \
  if (const auto error = expr) {                          \
    fmt::print(FMT_STRING("Error {}: {}\n"), err, error); \
    return __LINE__;                                      \
  }

void drawBitmap(cv::Mat& mat, FT_Bitmap& bitmap, int start_x, int start_y) {
  for (int y = 0; y < bitmap.rows; ++y) {
    auto draw_y = start_y + y;
    if (draw_y < 0 || draw_y >= mat.rows) {
      break;
    }

    for (int x = 0; x < bitmap.width; ++x) {
      auto draw_x = start_x + x;
      if (draw_x < 0 || draw_x >= mat.cols) {
        break;
      }

      mat.at<uint8_t>(draw_y, draw_x) = bitmap.buffer[y * bitmap.width + x];
    }
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_path == "") {
    fmt::print("provide a font path dipshit\n");
    return -1;
  } else {
    fmt::print("Font path: {}\n", FLAGS_font_path);
  }

  FT_Library library;
  CHECK_FT(FT_Init_FreeType(&library), "init ft2");

  // Figure out how many font faces are in the font file
  FT_Face face_counter;
  CHECK_FT(FT_New_Face(library, FLAGS_font_path.c_str(), -1, &face_counter),
           "counting font faces");

  for (int i = 0; i < face_counter->num_faces; ++i) {
    // Now actually load the faces, and set the font size to something decent
    FT_Face face;
    CHECK_FT(FT_New_Face(library, FLAGS_font_path.c_str(), i, &face),
             fmt::format("loading font face {}", i));
    // 50 pt at 100 dpi
    CHECK_FT(FT_Set_Char_Size(face, 72 * 64, 0, 100, 0), "set char size");

    cv::Mat mat(1000, 2000, CV_8UC1);
    std::string text = "The quick brown fox jumps over the lazy dog!";
    // std::string text = "foo";
    int px = 100;
    int py = 200;

    for (const char c : text) {
      CHECK_FT(FT_Load_Char(face, c, FT_LOAD_RENDER), "load char");

      auto slot = face->glyph;
      fmt::print("{} bitmap left: {}, top: {}\n", c, slot->bitmap_left,
                 slot->bitmap_top);
      auto& bitmap = slot->bitmap;
      fmt::print("{} bitmap rows: {}, bitmap width: {}\n", c, bitmap.rows,
                 bitmap.width);

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

      fmt::print("advancing x by {}, y by {}\n", slot->advance.x,
                 slot->advance.y);
      px += slot->advance.x >> 6;
      py += slot->advance.y >> 6;
    }

    cv::imshow("test", mat);
    cv::waitKey(0);

    FT_Done_Face(face);
  }

  FT_Done_Face(face_counter);
  FT_Done_FreeType(library);

  return 0;
}
