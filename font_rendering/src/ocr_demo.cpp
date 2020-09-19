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
#include <opencv2/imgproc.hpp>

#include "font_rendering/levenshtein.h"
#include "iosifovitch/iosifovitch.hpp"

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_bool(errors_only, false, "Show only images with errors");
DEFINE_string(atlas, "", "Override atlas");
DEFINE_bool(cells, true, "Put each character in its own cell");
DEFINE_int32(dpi, 300, "DPI to use for font rendering");
DEFINE_int32(point, 12, "Point size for font rendering");
DEFINE_int32(border, RendererSpacing::DEFAULT, "Border size around cells");
DEFINE_uint32(padding, 0, "Padding size around image");

inline static const std::vector<std::string> font_extensions{
    ".otf", ".ttf", ".svg", ".eot", ".woff", ".woff2", ".ttc"};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  // Initialize OCR
  auto tesseract = std::make_unique<tesseract::TessBaseAPI>();
  const auto tessdata_path = (fs::path(__FILE__).parent_path() / ".." / "data" / "tessdata-best").string();
  // if (tesseract->Init(NULL, "eng")) {
  if (tesseract->Init(tessdata_path.c_str(), "eng")) {
    // if (tesseract->Init("/data/fun_with_fonts/font_rendering/data/tessdata",
    // "eng")) { if
    // (tesseract->Init("/data/fun_with_fonts/font_rendering/data/tessdata-best",
    // "eng")) { if
    // (tesseract->Init("/data/fun_with_fonts/font_rendering/data/tessdata-fast",
    // "eng")) {
    fmt::print("Unable to initialize tesseract\n");
    return -2;
  }

  if (!tesseract->SetVariable("debug_file", "/dev/null")) {
    fmt::print("Unable to quiet tesseract\n");
    return -3;
  }

  RendererSpacing spacing(FLAGS_dpi, FLAGS_point, FLAGS_border);
  spacing.atlas_padding = spacing.em;
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

  const auto atlas = r.getAtlasString();

  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (fs::is_regular_file(p)) {
        // std::find(font_extensions.begin(), font_extensions.end(),
        //           p.path().extension()) != font_extensions.end()) {
      auto canonical = fs::canonical(p).string();
      // fmt::print("Loading font from: {}\n", canonical);
      r.loadFontFace(canonical);
      auto [mat, stats] = r.renderAtlas(FLAGS_cells);

      if (stats) {
        fmt::print("Issue while rendering font {}: {}\n", canonical, stats);
      } else {
        if (FLAGS_errors_only) {
          continue;
        }
      }

      if (mat.empty()) {
        continue;
      }

      std::cout << "Rendered from [" << stats.left_bound << ", "
                << stats.top_bound << "] to [" << stats.right_bound << ", "
                << stats.bottom_bound << "]. Image Size: [" << mat.cols << ", "
                << mat.rows << "]\n";

      cv::Mat tight_mat =
          mat(cv::Rect(stats.left_bound, stats.top_bound,
                       stats.right_bound - stats.left_bound + 1,
                       stats.bottom_bound - stats.top_bound + 1));
      cv::Mat inv(tight_mat.rows + 2 * FLAGS_padding,
                  tight_mat.cols + 2 * FLAGS_padding, CV_8UC1, 255);
      cv::Mat inv_roi = inv(cv::Rect(FLAGS_padding, FLAGS_padding,
                                     tight_mat.cols, tight_mat.rows));
      cv::subtract(static_cast<uint8_t>(255), tight_mat, inv_roi);

      tesseract->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
      tesseract->SetImage(inv.data, inv.cols, inv.rows, 1, inv.step);
      tesseract->SetSourceResolution(FLAGS_dpi);

      auto text = std::string(tesseract->GetUTF8Text());
      std::cout << "Reference: " << atlas;
      std::cout << "Detected: " << text;
      std::cout << "Levenshtein Distance: "
                << iosifovitch::levenshtein_distance(atlas, text) << "\n";
      std::cout << "Lower Levenshtein Distance: "
                << iosifovitch::levenshtein_distance(lowerAsciiInUtf8(atlas),
                                                     lowerAsciiInUtf8(text))
                << "\n";
      std::cout << "\n" << std::flush;

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
