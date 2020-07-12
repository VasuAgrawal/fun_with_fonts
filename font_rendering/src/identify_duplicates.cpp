// Recursively traverse through a directory tree, rendering fonts and moving the
// ones that fail some basic and deterministic checks to another directory.

#include <fmt/format.h>
#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>
#include <sstream>
#include <unordered_map>

#include "font_rendering/recursive_font_mapper.h"
#include "font_rendering/renderer.h"

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_string(show_images_for, "", "Show fonts that have the given duplicates");

struct alignas(64) DuplicateStats {
  std::unordered_map<std::string, uint32_t> match_set_counts;
  std::vector<cv::Mat> match_images;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  const auto atlas = makeFullAtlas();

  std::vector<Renderer> renderers;
  renderers.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    renderers.emplace_back(Renderer(atlas));
  }

  std::vector<DuplicateStats> duplicate_stats(FLAGS_thread_count);

  RecursiveFontMapper mapper(FLAGS_thread_count);
  mapper.runAndWait(
      [&](int32_t thread_index, const std::string& canonical) {
        auto& r = renderers[thread_index];
        r.loadFontFace(canonical);
        auto [mat, render_stats] = r.renderAtlas(true, FLAGS_show_images_for);
        const auto match_strings = RenderStats::makeMatchStrings(
            render_stats.matched_bitmaps);

        auto& stats = duplicate_stats[thread_index];
        for (const auto& s : match_strings) {
          if (s == FLAGS_show_images_for) {
            stats.match_images.emplace_back(mat.clone());
          }

          ++stats.match_set_counts[s];
        }
      },
      FLAGS_font_dir, FLAGS_count);

  DuplicateStats merged;
  // Merge all of the match_set_counts into a single one.
  for (const auto& stats : duplicate_stats) {
    for (const auto& [match_set, count] : stats.match_set_counts) {
      merged.match_set_counts[match_set] += count;
    }

    merged.match_images.insert(
        merged.match_images.end(),
        std::make_move_iterator(stats.match_images.begin()),
        std::make_move_iterator(stats.match_images.end()));
  }

  for (const auto& image : merged.match_images) {
    cv::imshow("Match image", image);
    if (cv::waitKey(0) == 'q') {
      break;
    }
  }

  fmt::print("Match set,Occurences,Length\n");
  for (auto [match_set, count] : merged.match_set_counts) {
    if (FLAGS_show_images_for != "" && FLAGS_show_images_for != match_set) {
      continue;
    }
    std::string escaped = match_set;
    auto pos = match_set.find('"');
    if (pos != std::string::npos) {
      escaped.replace(pos, 1, "\"\"");
    }
    // Need to surround match string in quotes
    // Need to add an extra ' to the beginning to force text mode in sheets
    // Adding a length column to avoid having to compute it in sheets
    fmt::print("\"'{}\",{},{}\n", escaped, count, match_set.size());
  }
}
