// Recursively traverse through a directory tree, rendering fonts and moving the
// ones that fail some basic and deterministic checks to another directory.

#include <filesystem>
#include <thread>
#include <unordered_map>
#include <sstream>

#include "font_rendering/recursive_font_mapper.h"
#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <folly/ProducerConsumerQueue.h>
#include <gflags/gflags.h>
#include <opencv2/highgui.hpp>

DEFINE_bool(verbose, false, "More detailed output");
DEFINE_bool(dry_run, false, "Don't actually move any files");
DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(error_dir, "", "Path to error directory");
DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_string(show_images_for, "", "Show fonts that have the given duplicates");

struct alignas(64) DuplicateStats {
  std::unordered_map<std::string, uint32_t> match_set_counts;
  std::vector<cv::Mat> match_images;
  uint32_t duplicate_alphabet = 0;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  } else if (FLAGS_error_dir == "") {
    fmt::print("provide an error dir dipshit\n");
    return -2;
  }

  // Make the error directories
  // fs::path error_dir(FLAGS_error_dir);
  // fs::create_directories(error_dir / MoveStats::FAILED_FONT_LOAD_DIR);
  // fs::create_directories(error_dir / MoveStats::FAILED_CHAR_LOAD_DIR);
  // fs::create_directories(error_dir / MoveStats::EMPTY_CHAR_DIR);

  // Atlas is composed of all of the printable ascii characters.
  const char first = ' ';
  const char last = '~';
  std::vector<std::string> atlas;
  {
    std::stringstream s;
    size_t written = 0;
    for (char c = first; c <= last; ++c) {
      s << c;
      ++written;
      if (written == 12) {
        atlas.emplace_back(s.str());
        s.str(std::string());
        written = 0;
      }
    }
    if (written) {
      atlas.emplace_back(s.str());
    }
  }

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

        std::vector<std::vector<char>> match_sets;

        for (const auto [a, b] : render_stats.matched_bitmaps) {
          bool a_found = false;
          bool b_found = false;
          // Try to find a match set that contains one of the characters.
          for (auto& set : match_sets) {
            for (const auto elem : set) {
              a_found |= elem == a;
              b_found |= elem == b;
            } 

            // If we've found a match for either character in this set, we'll
            // add the missing one and then stop searching sets.
            if (a_found) {
              set.push_back(b);
              break;
            }

            if (b_found) {
              set.push_back(a);
              break;
            }
          }

          if (!a_found && !b_found) {
            std::vector set{a, b};
            match_sets.emplace_back(std::move(set)); 
          }
        }

        auto& stats = duplicate_stats[thread_index];
        for (auto& set : match_sets) {
          std::sort(set.begin(), set.end());
          std::stringstream s;
          for (const auto elem : set) {
            s << elem;
          }

          const auto str = s.str();
          if (str == FLAGS_show_images_for) { // str can never be "
            stats.match_images.emplace_back(mat.clone());
          }
          ++stats.match_set_counts[s.str()];
        }
      },
      FLAGS_font_dir, FLAGS_count);


  DuplicateStats merged;
  // Merge all of the match_set_counts into a single one.
  for (const auto& stats : duplicate_stats) {
    for (const auto& [match_set, count] : stats.match_set_counts) {
      merged.match_set_counts[match_set] += count;
    }

    merged.match_images.insert(merged.match_images.end(),
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
