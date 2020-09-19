#include <filesystem>
#include <iostream>
#include <sstream>

#include "font_rendering/recursive_font_mapper.h"
#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_bool(errors_only, false, "Show only images with errors");
DEFINE_string(atlas, "", "Override atlas");
DEFINE_string(highlight, "", "Highlight characters");
DEFINE_bool(cells, true, "Put each character in its own cell");

// duplicated from recursive_font_mapper.h
inline static const std::vector<std::string> KNOWN_FONT_EXTENSIONS{
    ".otf", ".ttf", ".svg", ".eot", ".woff", ".woff2", ".ttc"};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  RendererSpacing spacing;
  // spacing.atlas_border = 0;
  Renderer r(spacing);

  if (FLAGS_atlas != "") {
    std::vector<std::string> user_atlas;
    std::istringstream stream(FLAGS_atlas);
    std::string line;
    while (std::getline(stream, line, '\n')) {
      user_atlas.push_back(line);
    }

    r = Renderer(user_atlas, spacing);
  }

  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (fs::is_regular_file(p) &&
        true) {
        // std::find(KNOWN_FONT_EXTENSIONS.begin(), KNOWN_FONT_EXTENSIONS.end(),
        //           p.path().extension()) != KNOWN_FONT_EXTENSIONS.end()) {
      auto canonical = fs::canonical(p).string();
      // fmt::print("Loading font from: {}\n", canonical);
      r.loadFontFace(canonical);
      auto [mat, err] = r.renderAtlas(FLAGS_cells, FLAGS_highlight);
      if (err) {
        fmt::print("Issue while rendering font {}: {}\n", canonical, err);
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
