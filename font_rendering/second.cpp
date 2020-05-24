#include "renderer.h"

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_dir, "", "Path to font directory");

inline static const std::vector<std::string> font_extensions{".otf", ".ttf"};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  Renderer r;

  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (fs::is_regular_file(p) &&
        std::find(font_extensions.begin(), font_extensions.end(),
                  p.path().extension()) != font_extensions.end()) {
      auto canonical = fs::canonical(p).string();
      fmt::print("Loading font from: {}\n", canonical);
      r.loadFontFace(canonical);
      auto mat = r.renderAtlas();

      if (!mat) {
        fmt::print("Unable to render font {}\n", canonical);
      } else {
        cv::imshow("Rendered", mat.value());
        if (cv::waitKey(0) == 'q') {
          break;
        }
      }
    }
  }
}
