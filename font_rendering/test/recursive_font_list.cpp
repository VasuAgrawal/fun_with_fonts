// Simple binary to test recursive font mapper.

#include <fmt/format.h>
#include <gflags/gflags.h>

#include "font_rendering/recursive_font_mapper.h"

DEFINE_string(font_dir, "", "Path to font directory");
DEFINE_int32(thread_count, 24, "Number of render threads to use");
DEFINE_int32(count, 0, "Number of files to stop after");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_font_dir == "") {
    fmt::print("provide a font dir dipshit\n");
    return -1;
  }

  RecursiveFontMapper mapper(FLAGS_thread_count);
  mapper.runAndWait(
      [](int32_t thread_index, const std::string& canonical) {
        fmt::print("[Thread {}] {}\n", thread_index, canonical);
      },
      FLAGS_font_dir, FLAGS_count);
}
