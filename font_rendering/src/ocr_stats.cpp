#include <tesseract/publictypes.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>

#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <tesseract/baseapi.h>

#include <opencv2/highgui.hpp>

#include "font_rendering/levenshtein.h"
#include "font_rendering/recursive_font_mapper.h"
#include "iosifovitch/iosifovitch.hpp"

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(atlas, "", "Override atlas");
DEFINE_int32(dpi, 300, "DPI to use for font rendering");
DEFINE_int32(point, 12, "Point size for font rendering");
DEFINE_uint32(padding, 0, "Padding size around image");
DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_bool(lowercase, true, "Use lower case levenshtein distance");
DEFINE_bool(csv, false, "Output text in CSV format");

template <typename T>
class Counter : public std::map<T, T> {  // we care about sorted order
 public:
  Counter<T>& operator+=(const Counter<T>& other) {
    for (const auto& [k, v] : other) {
      (*this)[k] += v;
    }

    return *this;
  }
};

struct alignas(64) OcrStats {
  Counter<size_t> levenshtein;
  Counter<size_t> levenshtein_lowercase;
  Counter<size_t> levenshtein_delta;
  size_t skipped = 0;

  OcrStats& operator+=(const OcrStats& other) {
    levenshtein += other.levenshtein;
    levenshtein_lowercase += other.levenshtein_lowercase;
    levenshtein_delta += other.levenshtein_delta;
    skipped += other.skipped;

    return *this;
  }
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  RecursiveFontMapper mapper(FLAGS_thread_count);

  // Initialize OCR
  std::vector<std::unique_ptr<tesseract::TessBaseAPI>> tesseracts;
  tesseracts.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    auto& tesseract =
        tesseracts.emplace_back(std::make_unique<tesseract::TessBaseAPI>());
    // if (tesseract->Init(NULL, "eng")) {
    if (tesseract->Init(
          "/data/fun_with_fonts/font_rendering/data/tessdata-best", "eng")) {
      fmt::print("Unable to initialize tesseract\n");
      return -2;
    }

    if (!tesseract->SetVariable("debug_file", "/dev/null")) {
      fmt::print("Unable to quiet tesseract\n");
      return -3;
    }
  }

  RendererSpacing spacing(FLAGS_dpi, FLAGS_point);
  spacing.atlas_padding = spacing.em;
  std::vector<Renderer> renderers;
  renderers.reserve(FLAGS_thread_count);

  std::vector<std::string> user_atlas;
  if (FLAGS_atlas != "") {
    std::istringstream stream(FLAGS_atlas);
    std::string line;
    while (std::getline(stream, line, '\n')) {
      user_atlas.push_back(line);
    }
  }

  for (int i = 0; i < FLAGS_thread_count; ++i) {
    if (FLAGS_atlas != "") {
      renderers.emplace_back(user_atlas, spacing);
    } else {
      renderers.emplace_back(spacing);
    }
  }

  std::vector<OcrStats> ocr_stats(FLAGS_thread_count);

  mapper.runAndWait(
      [&](int32_t thread_index, const std::string& canonical) {
        auto& r = renderers[thread_index];
        r.loadFontFace(canonical);
        auto [mat, err] = r.renderAtlas(false);

        auto& stats = ocr_stats[thread_index];
        // if (err) {
        // ++stats.skipped;
        // return;
        // }

        if (mat.empty()) {
          ++stats.skipped;
          return;
        }

        // cv::Mat inv;
        // cv::subtract(static_cast<uint8_t>(255), mat, inv);

        cv::Mat tight_mat = mat(cv::Rect(err.left_bound, err.top_bound,
                                        err.right_bound - err.left_bound + 1,
                                        err.bottom_bound - err.top_bound + 1));
        cv::Mat inv(tight_mat.rows + 2 * FLAGS_padding,
                    tight_mat.cols + 2 * FLAGS_padding, CV_8UC1, 255);
        cv::Mat inv_roi = inv(
            cv::Rect(FLAGS_padding, FLAGS_padding, tight_mat.cols, tight_mat.rows));
        cv::subtract(static_cast<uint8_t>(255), tight_mat, inv_roi);

        // cv::imshow("Rendered", inv);
        // cv::waitKey(0);

        auto& tesseract = tesseracts[thread_index];
        // tesseract->SetPageSegMode(tesseract::PSM_AUTO);
        tesseract->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
        tesseract->SetImage(inv.data, inv.cols, inv.rows, 1, inv.step);
        tesseract->SetSourceResolution(FLAGS_dpi);

        const auto text = std::string(tesseract->GetUTF8Text());
        const auto atlas = r.getAtlasString();
        const auto distance = iosifovitch::levenshtein_distance(atlas, text);
        const auto lowercase_distance = iosifovitch::levenshtein_distance(
            lowerAsciiInUtf8(atlas), lowerAsciiInUtf8(text));
        ++stats.levenshtein[distance];
        ++stats.levenshtein_lowercase[lowercase_distance];
        ++stats.levenshtein_delta[distance - lowercase_distance];
      },
      FLAGS_font_dir, FLAGS_count

  );

  for (auto& t : tesseracts) {
    t->End();
  }

  OcrStats total_ocr_stats;
  for (const auto& s : ocr_stats) {
    total_ocr_stats += s;
  }

  const auto lower_str = FLAGS_lowercase ? "Lowercase Levenshtein" : "Levenshtein";

  // Print appropriate header
  if (FLAGS_csv) {
    fmt::print("{} Distance,Count\n", lower_str);
  } else {
    fmt::print("{} Distance Counts\n", lower_str);
  }

  const auto& counts = FLAGS_lowercase ? total_ocr_stats.levenshtein_lowercase : total_ocr_stats.levenshtein;

  for (const auto& [k, v] : counts) {
    if (FLAGS_csv) {
      fmt::print("{},{}\n", k, v);
    } else {
      fmt::print("{}: {}\n", k, v);
    }
  }

  if (FLAGS_csv) {
    fmt::print("Skipped,{}\n", total_ocr_stats.skipped);
  } else {
    fmt::print("Skipped: {}\n", total_ocr_stats.skipped);
  }
}
