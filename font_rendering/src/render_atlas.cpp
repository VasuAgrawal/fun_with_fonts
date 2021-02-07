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
DEFINE_int32(offset, RendererSpacing::DEFAULT, "Bottom offset for font rendering");
DEFINE_uint32(write_style, 0, "Image write style (0-3)");

DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_bool(stats, false, "Output stats in JSON format");
DEFINE_bool(save, true, "Save images to specified output folder");
DEFINE_bool(skip_cell_out, true, "Skip writing images which fall outside their cell");

static constexpr size_t QUEUE_SIZE = 10;

struct alignas(64) Stats {
  uint32_t out_of_image_bounds = 0;
  uint32_t out_of_cell_bounds = 0;
  uint32_t image_count = 0;
  uint32_t empty_cell = 0;

  Stats& operator+=(const Stats& other) {
    out_of_image_bounds += other.out_of_image_bounds;
    out_of_cell_bounds += other.out_of_cell_bounds;
    image_count += other.image_count;
    empty_cell += other.empty_cell;

    return *this;
  }
};

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
  if (FLAGS_save) {
    fs::create_directories(output_dir);
  }

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

  RendererSpacing spacing(FLAGS_dpi, FLAGS_point, FLAGS_border, FLAGS_padding,
      FLAGS_offset);

  std::vector<Renderer> renderers;
  renderers.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    renderers.emplace_back(Renderer(atlas, spacing));
  }

  RecursiveFontMapper mapper(FLAGS_thread_count);

  std::vector<Stats> stats(FLAGS_thread_count);

  mapper.runAndWait(
      [&](int32_t thread_index, const std::string& canonical) {
        auto& r = renderers[thread_index];
        r.loadFontFace(canonical);
        auto [mat, err] = r.renderAtlas();
        auto& stat = stats[thread_index];
        stat.out_of_image_bounds += !!err.out_of_image_bounds_count;
        stat.out_of_cell_bounds += !!err.out_of_cell_bounds_count;
        ++stat.image_count;
        stat.empty_cell += !err.empty_characters.empty();

        if (FLAGS_skip_cell_out && err.out_of_cell_bounds_count) {
          return; 
        }

        if (FLAGS_save) {
          const auto output_basename = fs::path(canonical).filename().string();
          r.saveImage(output_dir, output_basename, mat,
                      static_cast<ImageWriteStyle>(FLAGS_write_style));
        }
      },
      FLAGS_font_dir, FLAGS_count);

  Stats total_stats;
  for (size_t i = 0; const auto& s : stats) {
    total_stats += s;
    // fmt::print("Thread {}: {{ \"total\": {}, \"out_of_image_bounds\": {}, \"out_of_cell_bounds\": {} }}\n",
    //     i++, s.image_count, s.out_of_image_bounds,
    //     s.out_of_cell_bounds);
  }

  if (FLAGS_stats) {
  fmt::print("{{ \"total\": {}, \"out_of_image_bounds\": {}, \"out_of_cell_bounds\": {}, \"empty_cell\": {} }}\n",
      total_stats.image_count, total_stats.out_of_image_bounds,
      total_stats.out_of_cell_bounds,
      total_stats.empty_cell);
  }
}
