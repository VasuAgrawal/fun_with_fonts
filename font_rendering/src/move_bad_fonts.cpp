// Recursively traverse through a directory tree, rendering fonts and moving the
// ones that fail some basic and deterministic checks to another directory.

#include <filesystem>
#include <thread>

#include "font_rendering/recursive_font_mapper.h"
#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

#include <fmt/format.h>
#include <folly/ProducerConsumerQueue.h>
#include <gflags/gflags.h>

DEFINE_bool(verbose, false, "More detailed output");
DEFINE_bool(dry_run, false, "Don't actually move any files");
DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(error_dir, "", "Path to error directory");
DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");

static constexpr size_t QUEUE_SIZE = 100;

// This should be basically all of the symbols you could want to care about,
// which will help filter the _really_ good fonts from the okay fonts.

// 64 byte alignment seems to be the case on my machine at least.
// grep "cache_alignment" /proc/cpuinfo
struct alignas(64) MoveStats {
  inline static const std::string FAILED_FONT_LOAD_DIR = "failed_font_loads";
  inline static const std::string FAILED_CHAR_LOAD_DIR = "failed_char_loads";
  inline static const std::string EMPTY_CHAR_DIR = "empty_characters";

  uint32_t failed_font_loads = 0;
  uint32_t failed_char_loads = 0;
  uint32_t empty_characters = 0;

  MoveStats& operator+=(const MoveStats& other) {
    failed_font_loads += other.failed_font_loads;
    failed_char_loads += other.failed_char_loads;
    empty_characters += other.empty_characters;
    return *this;
  }
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
  fs::path error_dir(FLAGS_error_dir);
  fs::create_directories(error_dir / MoveStats::FAILED_FONT_LOAD_DIR);
  fs::create_directories(error_dir / MoveStats::FAILED_CHAR_LOAD_DIR);
  fs::create_directories(error_dir / MoveStats::EMPTY_CHAR_DIR);

  // Atlas is composed of all of the printable ascii characters, except space.
  const char first = ' ';
  const char last = '~';
  std::vector<std::string> atlas;
  atlas.reserve(last - first + 1);
  for (char c = first; c <= last; ++c) {
    atlas.emplace_back(1, c);
  }

  RecursiveFontMapper mapper(FLAGS_thread_count);

  std::vector<Renderer> renderers;
  renderers.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    renderers.emplace_back(Renderer(atlas));
  }
  std::vector<MoveStats> move_stats(FLAGS_thread_count);

  mapper.runAndWait(
      [&](int32_t thread_index, const std::string& canonical) {
        auto& r = renderers[thread_index];
        r.loadFontFace(canonical);
        auto [mat, render_stats] = r.renderAtlas();

        auto& m = move_stats[thread_index];
        fs::path canonical_path = canonical;

        if (render_stats.font_not_loaded) {
          // Font load failed for this font, it's never going to get better.
          const auto target =
              fs::canonical(error_dir / MoveStats::FAILED_FONT_LOAD_DIR) /
              canonical_path.filename();

          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
            fmt::print("Couldn't load font. Moving from {} to {}.\n", canonical,
                       target.string());
          }

          ++m.failed_font_loads;
          return;
        }

        if (render_stats.char_loads_failed.size()) {
          const auto target =
              fs::canonical(error_dir / MoveStats::FAILED_CHAR_LOAD_DIR) /
              canonical_path.filename();

          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
            fmt::print("char load failed for font. Moving from {} to {}.\n",
                       canonical, target.string());
          }

          ++m.failed_char_loads;
          return;
        }

        if (render_stats.empty_characters.size()) {
          const auto target =
              fs::canonical(error_dir / MoveStats::EMPTY_CHAR_DIR) /
              canonical_path.filename();

          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
            // Nonzero empty characters means the font is incomplete, we
            // definitely don't want it.
            fmt::print(
                "Missing some characters in font. Moving from {} to {}.\n",
                canonical, target.string());
          }

          ++m.empty_characters;
          return;
        }
      },
      FLAGS_font_dir, FLAGS_count);

  // Now, we should be able to count all of the stats.
  MoveStats total_move_stats;
  for (const auto& s : move_stats) {
    total_move_stats += s;
  }

  fmt::print("Fonts with failed font loads: {}\n",
             total_move_stats.failed_font_loads);
  fmt::print("Fonts with failed char loads: {}\n",
             total_move_stats.failed_char_loads);
  fmt::print("Fonts with empty characters: {}\n",
             total_move_stats.empty_characters);
}
