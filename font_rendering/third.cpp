#include "renderer.h"

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(output_dir, "", "Path to save output images");

inline static const std::vector<std::string> font_extensions{".otf", ".ttf"};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  } else if (FLAGS_output_dir == "") {
    fmt::print("provide an output dir dipshit\n");
    return -1;
  }

  fs::path output_dir(FLAGS_output_dir);
  fs::create_directories(output_dir);

  Renderer r;

  int count = 0;
  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (FLAGS_count > 0 && count >= FLAGS_count) {
      return 0;
    }

    if (fs::is_regular_file(p) &&
        std::find(font_extensions.begin(), font_extensions.end(),
                  p.path().extension()) != font_extensions.end()) {

      auto canonical = fs::canonical(p).string();

      if (!r.loadFontFace(canonical)) {
        fmt::print("Unable to load font from: {}\n", canonical);
        continue;
      }
      
      auto mat = r.renderAtlas();

      if (!mat) {
        fmt::print("Unable to render font {}\n", canonical);
        continue;
      } 
      
      auto output = output_dir / (p.path().stem().string() + std::string(".png"));
      cv::imwrite(output.string(), mat.value());

      ++count;
    }
  }
}
