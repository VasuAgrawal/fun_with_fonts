// Recursively traverse through a directory tree, rendering fonts and moving the
// ones that fail some basic and deterministic checks to another directory.

#include <fmt/format.h>
#include <gflags/gflags.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
namespace fs = std::filesystem;

#include "font_rendering/recursive_font_mapper.h"
#include "font_rendering/renderer.h"

DEFINE_bool(verbose, false, "More detailed output");
DEFINE_bool(dry_run, false, "Don't actually move any files");
DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(error_dir, "", "Path to error directory");
DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_string(whitelist, "", "Path to .txt with allowed matches");

struct alignas(64) DuplicateStats {
  inline static const std::string ALLOWED_DUPLICATE_DIR = "allowed_duplicates";
  inline static const std::string FAILED_DUPLICATE_DIR = "failed_duplicates";

  uint32_t allowed_duplicates = 0;
  uint32_t failed_duplicates = 0;

  DuplicateStats& operator+=(const DuplicateStats& other) {
    allowed_duplicates += other.allowed_duplicates;
    failed_duplicates += other.failed_duplicates;
    return *this;
  }
};

std::unordered_set<std::string> loadWhitelist(fs::path path) {
  if (path.empty()) {
    return {};
  }

  if (!fs::exists(path)) {
    fmt::print("Warning: whitelist at {} doesn't exist!", path.string());
    return {};
  }

  if (!fs::is_regular_file(path)) {
    fmt::print("Warning: whitelist at {} isn't a file!", path.string());
    return {};
  }

  std::unordered_set<std::string> whitelist;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    whitelist.insert(line);
  }

  return whitelist;
}

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
  fs::create_directories(error_dir / DuplicateStats::ALLOWED_DUPLICATE_DIR);
  fs::create_directories(error_dir / DuplicateStats::FAILED_DUPLICATE_DIR);

  const auto whitelist = loadWhitelist(FLAGS_whitelist);
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
        auto [mat, render_stats] = r.renderAtlas();
        const auto match_strings =
            RenderStats::makeMatchStrings(render_stats.matched_bitmaps);

        // If there are no duplicates, don't do anything.
        if (render_stats.matched_bitmaps.empty()) {
          return;
        }

        // Figure out how many of the match strings are in the whitelist.
        size_t allowed_match_strings = 0;
        for (const auto& s : match_strings) {
          if (whitelist.contains(s)) {
            ++allowed_match_strings;
          }
        }

        fs::path canonical_path = canonical;

        if (match_strings.size() == allowed_match_strings) {
          // If all matches fall in the whitelist, move into a whitelist dir.
          const auto target =
              fs::canonical(error_dir / DuplicateStats::ALLOWED_DUPLICATE_DIR) /
              canonical_path.filename();

          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
            fmt::print("Font has allowed duplicates. Moving from {} to {}.\n",
                       canonical, target.string());
          }

          return;
        } else {
          // At least one match is not in the whitelist, move into another dir.

          const auto target =
              fs::canonical(error_dir / DuplicateStats::FAILED_DUPLICATE_DIR) /
              canonical_path.filename();

          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
            fmt::print("Font has bad duplicates. Moving from {} to {}.\n",
                       canonical, target.string());
          }
        }
      },
      FLAGS_font_dir, FLAGS_count);
}
