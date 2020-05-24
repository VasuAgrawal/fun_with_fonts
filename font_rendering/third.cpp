#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

#include "renderer.h"
namespace fs = std::filesystem;

#include <folly/ProducerConsumerQueue.h>
#include <gflags/gflags.h>

#include <opencv2/highgui.hpp>

DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");
DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_string(output_dir, "", "Path to save output images");

inline static const std::vector<std::string> font_extensions{
  ".otf", ".ttf", ".svg", ".eot", ".woff", ".woff2"};
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

  // Create the render queues
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<std::string>>>
      render_queues;
  render_queues.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    render_queues.emplace_back(
        std::make_unique<folly::ProducerConsumerQueue<std::string>>(
            QUEUE_SIZE));
  }

  fs::path output_dir(FLAGS_output_dir);
  fs::create_directories(output_dir);
  std::atomic_bool new_work{true};

  // Create the renderer threads
  std::vector<std::thread> render_threads;
  render_threads.reserve(FLAGS_thread_count);
  for (int i = 0; i < FLAGS_thread_count; ++i) {
    render_threads.emplace_back([&new_work, &render_queues, i, output_dir]() {
      Renderer r;
      std::string canonical;
      const std::vector<int> compression_params{cv::IMWRITE_PNG_COMPRESSION, 9};
      while (true) {
        if (!render_queues[i]->read(canonical)) {          // failed to get work
          if (new_work.load(std::memory_order_acquire)) {  // more to come
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1ms);
            continue;
          } else {  // Nothing else is coming, we're done
            break;
          }
        }

        if (!r.loadFontFace(canonical)) {
          fmt::print("Unable to load font from: {}\n", canonical);
          continue;
        }

        auto mat = r.renderAtlas();

        if (!mat) {
          fmt::print("Unable to render font {}\n", canonical);
          continue;
        }

        auto output = output_dir / (fs::path(canonical).stem().string() +
                                    std::string(".png"));
        cv::imwrite(output.string(), mat.value(), compression_params);
      }
    });
  }

  int count = 0;
  for (auto& p : fs::recursive_directory_iterator(FLAGS_font_dir)) {
    if (FLAGS_count > 0 && count >= FLAGS_count) {
      break;
    }

    // Only consider regular files that have the right extension
    if (!fs::is_regular_file(p) ||
        std::find(font_extensions.begin(), font_extensions.end(),
                  p.path().extension()) == font_extensions.end()) {
      continue;
    }

    ++count;

    // const auto canonical = fs::canonical(p).string();
    const auto canonical = p.path().string();

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
}