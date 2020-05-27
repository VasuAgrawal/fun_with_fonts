#include "renderer.h"

#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;

#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_bool(errors_only, false, "Show only images with errors");

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
      // fmt::print("Loading font from: {}\n", canonical);
      r.loadFontFace(canonical);
      auto [mat, err] = r.renderAtlas();

      if (auto e = static_cast<int>(err); e) {
        fmt::print("Issue while rendering font {}: {}\n", canonical, RendererErrorStrings[e]);
      } else {
        if (FLAGS_errors_only) {
          continue;
        }
      }

      if (!mat.empty()) {
        cv::imshow("Rendered", mat);
        if (cv::waitKey(0) == 'q') {
          break;
        }
      }

    }
  }
}
