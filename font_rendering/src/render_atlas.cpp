#include <atomic>
#include <chrono>
#include <thread>

#include "font_rendering/recursive_font_mapper.h"
#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <folly/ProducerConsumerQueue.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/highgui.hpp>

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(output_dir, "", "Path to save output images");
DEFINE_string(atlas, "", "Override atlas");

DEFINE_int32(dpi, 300, "DPI to use for font rendering");
DEFINE_int32(point, 12, "Point size for font rendering");
DEFINE_int32(border, RendererSpacing::DEFAULT, "Border for font rendering");
DEFINE_int32(padding, RendererSpacing::DEFAULT, "Padding for font rendering");
DEFINE_uint32(write_style, 0, "Image write style (0-3)");

DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");

static constexpr size_t QUEUE_SIZE = 10;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  } else if (FLAGS_output_dir == "") {
    fmt::print("provide an output dir dipshit\n");
    return -1;
  }

  if (FLAGS_write_style >= static_cast<uint64_t>(ImageWriteStyle::COUNT)) {
    fmt::print("Invalid write style {} specified\n", FLAGS_write_style);
    return -1;
  }

  FLAGS_log_dir = FLAGS_output_dir;
  google::InitGoogleLogging(argv[0]);

  fs::path output_dir(FLAGS_output_dir);
  fs::create_directories(output_dir);

  auto atlas = makeFullAtlas();
  if (FLAGS_atlas != "") {
    std::vector<std::string> user_atlas;
    std::istringstream stream(FLAGS_atlas);
    std::string line;
    while (std::getline(stream, line, '\n')) {
      user_atlas.push_back(line);
    }

    atlas = user_atlas;
  }

  RendererSpacing spacing(FLAGS_dpi, FLAGS_point, FLAGS_border, FLAGS_padding);

  std::vector<Renderer> renderers;
  renderers.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    renderers.emplace_back(Renderer(atlas, spacing));
  }

  RecursiveFontMapper mapper(FLAGS_thread_count);
  mapper.runAndWait(
      [&](int32_t thread_index, const std::string& canonical) {
        auto& r = renderers[thread_index];
        r.loadFontFace(canonical);
        auto [mat, err] = r.renderAtlas();
        const auto output_basename = fs::path(canonical).filename().string();
        r.saveImage(output_dir, output_basename, mat,
                    static_cast<ImageWriteStyle>(FLAGS_write_style));
      },
      FLAGS_font_dir, FLAGS_count);
}
