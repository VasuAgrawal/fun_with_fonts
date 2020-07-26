#include <tesseract/publictypes.h>

#include <filesystem>
#include <iostream>
#include <sstream>

#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <tesseract/baseapi.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_bool(errors_only, false, "Show only images with errors");
DEFINE_string(atlas, "", "Override atlas");
DEFINE_bool(cells, true, "Put each character in its own cell");
DEFINE_int32(dpi, 300, "DPI to use for font rendering");
DEFINE_int32(point, 12, "Point size for font rendering");

inline static const std::vector<std::string> font_extensions{".otf", ".ttf"};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  // Initialize OCR
  auto tesseract = std::make_unique<tesseract::TessBaseAPI>();
  if (tesseract->Init(NULL, "eng")) {
    fmt::print("Unable to initialize tesseract\n");
    return -2;
  }

  Renderer r(RendererSpacing(FLAGS_dpi, FLAGS_point));

  if (FLAGS_atlas != "") {
    std::vector<std::string> user_atlas;
    std::istringstream stream(FLAGS_atlas);
    std::string line;
    while (std::getline(stream, line, '\n')) {
      user_atlas.push_back(line);
    }

    r = Renderer(user_atlas, RendererSpacing(FLAGS_dpi, FLAGS_point));
  }

  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (fs::is_regular_file(p) &&
        std::find(font_extensions.begin(), font_extensions.end(),
                  p.path().extension()) != font_extensions.end()) {
      auto canonical = fs::canonical(p).string();
      // fmt::print("Loading font from: {}\n", canonical);
      r.loadFontFace(canonical);
      auto [mat, err] = r.renderAtlas(FLAGS_cells);

      if (err) {
        fmt::print("Issue while rendering font {}: {}\n", canonical, err);
      } else {
        if (FLAGS_errors_only) {
          continue;
        }
      }

      if (mat.empty()) {
        continue;
      }

      cv::Mat inv;
      cv::subtract(static_cast<uint8_t>(255), mat, inv);

      tesseract->SetPageSegMode(tesseract::PSM_AUTO);
      tesseract->SetImage(inv.data, inv.cols, inv.rows, 1, inv.step);
      tesseract->SetSourceResolution(FLAGS_dpi);

      auto text = std::string(tesseract->GetUTF8Text());
      std::cout << text << std::flush;

      // auto text = std::unique_ptr<char[]>(tesseract->GetUTF8Text());
      // fmt::print("Recognized {}\n", text.get());

      cv::imshow("Rendered", inv);
      if (cv::waitKey(0) == 'q') {
        break;
      }
    }
  }

  tesseract->End();
}
