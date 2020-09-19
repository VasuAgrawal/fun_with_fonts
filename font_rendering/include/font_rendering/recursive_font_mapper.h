
#include <chrono>
#include <filesystem>
#include <thread>
namespace fs = std::filesystem;

#include <folly/ProducerConsumerQueue.h>

class RecursiveFontMapper {
  static constexpr size_t QUEUE_SIZE = 100;
  inline static const std::vector<std::string> KNOWN_FONT_EXTENSIONS{
      ".otf", ".ttf", ".svg", ".eot", ".woff", ".woff2", ".ttc"};

 public:
  explicit RecursiveFontMapper(int32_t thread_count = 48)
      : thread_count_(thread_count) {
    work_queues_.reserve(thread_count_);
    for (auto i = 0; i < thread_count_; ++i) {
      work_queues_.emplace_back(
          std::make_unique<folly::ProducerConsumerQueue<std::string>>(
              QUEUE_SIZE));
    }
  }

  template <typename Callable>
  void runAndWait(const Callable& fn, fs::path font_dir,
                  int32_t max_count = 0) {
    std::vector<std::thread> work_threads;
    work_threads.reserve(thread_count_);
    for (int i = 0; i < thread_count_; ++i) {
      work_threads.emplace_back(

          [&, i, &queue = work_queues_[i]]() {
            std::string canonical;

            while (true) {
              if (!queue->read(canonical)) {  // failed to get work
                if (new_work_.load(
                        std::memory_order_acquire)) {  // more to come
                  using namespace std::chrono_literals;
                  std::this_thread::sleep_for(1ms);
                  continue;
                } else {  // Nothing else is coming, we're done
                  break;
                }
              }

              fn(i, canonical);
            }
          });
    }

    int32_t count = 0;
    for (auto& p : fs::recursive_directory_iterator(font_dir)) {
      if (max_count > 0 && count >= max_count) {
        break;
      }

      if (!fs::is_regular_file(p)) {
      // ||
          // std::find(KNOWN_FONT_EXTENSIONS.begin(), KNOWN_FONT_EXTENSIONS.end(),
          //           p.path().extension()) == KNOWN_FONT_EXTENSIONS.end()) {
        continue;
      }

      ++count;
      const auto canonical = fs::canonical(p).string();

      // Use the count to pick a queue to add the path to. If that one's full,
      // iterate through the rest of them. If they're all full, wait a second,
      // and repeat, ad infineum.
      while (true) {
        // Try all the queues once, starting with the one we're supposed to
        for (int i = 0; i < work_queues_.size(); ++i) {
          const auto queue_index = (count + i) % work_queues_.size();
          if (work_queues_[queue_index]->write(canonical)) {
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

    new_work_.store(false, std::memory_order_release);
    for (int i = 0; i < work_threads.size(); ++i) {
      if (work_threads[i].joinable()) {
        work_threads[i].join();
      }
    }
  }

 private:
  int32_t thread_count_ = 0;
  std::atomic_bool new_work_{true};
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<std::string>>>
      work_queues_;
};
