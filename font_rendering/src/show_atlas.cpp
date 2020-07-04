#include <filesystem>
#include <iostream>
#include <sstream>

#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_bool(errors_only, false, "Show only images with errors");
DEFINE_string(atlas, "", "Override atlas");
DEFINE_bool(cells, true, "Put each character in its own cell");

inline static const std::vector<std::string> font_extensions{".otf", ".ttf"};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }
 
  Renderer r;

  if (FLAGS_atlas != "") {
    std::vector<std::string> user_atlas;
    std::istringstream stream(FLAGS_atlas);
    std::string line;
    while (std::getline(stream, line, '\n')) {
      user_atlas.push_back(line);
    }
 
    r = Renderer(user_atlas);
  }

  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (fs::is_regular_file(p) &&
        std::find(font_extensions.begin(), font_extensions.end(),
                  p.path().extension()) != font_extensions.end()) {
      auto canonical = fs::canonical(p).string();
      // fmt::print("Loading font from: {}\n", canonical);
      r.loadFontFace(canonical);
      auto [mat, err] = r.renderAtlas(FLAGS_cells);

      if (auto e = static_cast<int>(err); e) {
        fmt::print("Issue while rendering font {}: {}\n", canonical,
                   RendererErrorStrings[e]);
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
