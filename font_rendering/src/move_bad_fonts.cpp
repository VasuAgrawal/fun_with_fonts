// Recursively traverse through a directory tree, rendering fonts and moving the
// ones that fail some basic and deterministic checks to another directory.

#include <filesystem>
#include <thread>

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

  std::atomic_bool new_work{true};
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<std::string>>>
      render_queues;
  render_queues.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    render_queues.emplace_back(
        std::make_unique<folly::ProducerConsumerQueue<std::string>>(
            QUEUE_SIZE));
  }
  std::vector<MoveStats> move_stats(FLAGS_thread_count);

  std::vector<std::thread> render_threads;
  render_threads.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    render_threads.emplace_back([&, &queue = render_queues[i],
                                 &move_stats = move_stats[i]]() {
      Renderer r(atlas);

      std::string canonical;
      fs::path canonical_path;
      while (true) {
        if (!queue->read(canonical)) {                     // failed to get work
          if (new_work.load(std::memory_order_acquire)) {  // more to come
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1ms);
            continue;
          } else {  // Nothing else is coming, we're done
            break;
          }
        }

        // Have some actual work now
        r.loadFontFace(canonical);
        auto [mat, render_stats] = r.renderAtlas();

        canonical_path = canonical;

        if (render_stats.font_not_loaded) {
          // Font load failed for this font, it's never going to get better.
          const auto target = fs::canonical(error_dir / MoveStats::FAILED_FONT_LOAD_DIR) /
                              canonical_path.filename();

          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
            fmt::print("Couldn't load font. Moving from {} to {}.\n", canonical,
                      target.string());
          }
          
          ++move_stats.failed_font_loads;
          continue;
        }

        if (render_stats.char_loads_failed.size()) {
          const auto target = fs::canonical(error_dir / MoveStats::FAILED_CHAR_LOAD_DIR) /
                              canonical_path.filename();
          
          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
          fmt::print("char load failed for font. Moving from {} to {}.\n",
                     canonical, target.string());
          }

          ++move_stats.failed_char_loads;
          continue;
        }

        if (render_stats.empty_characters.size()) {
          const auto target =
              fs::canonical(error_dir / MoveStats::EMPTY_CHAR_DIR) / canonical_path.filename();
          
          if (!FLAGS_dry_run) {
            fs::rename(canonical_path, target);
          }

          if (FLAGS_verbose) {
          // Nonzero empty characters means the font is incomplete, we
          // definitely don't want it.
          fmt::print("Missing some characters in font. Moving from {} to {}.\n",
                     canonical, target.string());
          }

          ++move_stats.empty_characters;
          continue;
        }
      }
    });
  }

  size_t count = 0;
  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (FLAGS_count > 0 && count >= FLAGS_count) {
      break;
    }

    if (!fs::is_regular_file(p) ||
        std::find(KNOWN_FONT_EXTENSIONS.begin(), KNOWN_FONT_EXTENSIONS.end(),
                  p.path().extension()) == KNOWN_FONT_EXTENSIONS.end()) {
      continue;
    }

    ++count;
    const auto canonical = fs::canonical(p).string();
    // fmt::print("Loading font from: {}\n", canonical);

    // Use the count to pick a queue to add the path to. If that one's full,
    // iterate through the rest of them. If they're all full, wait a second,
    // and repeat, ad infineum.
    while (true) {
      // Try all the queues once, starting with the one we're supposed to
      for (int i = 0; i < render_queues.size(); ++i) {
        const auto queue_index = (count + i) % render_queues.size();
        if (render_queues[queue_index]->write(canonical)) {
          goto loaded;
        }
      }

      // And if we're not able to put a thing into a queue after cycling
      // through them all once, take a breather.
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1ms);
    }

  loaded:;
  }

  new_work.store(false, std::memory_order_release);
  for (int i = 0; i < render_threads.size(); ++i) {
    if (render_threads[i].joinable()) {
      render_threads[i].join();
    }
  }

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
