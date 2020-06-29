#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include <filesystem>
#include <opencv2/imgcodecs.hpp>

#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

std::string FONT_PATH =
    "/data/fun_with_fonts/font_rendering/test_fonts/ttf/0000000.ttf";

static void FT2_Init(benchmark::State& state) {
  for (auto _ : state) {
    Renderer r;
    // benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(FT2_Init);

static void LoadFontFace(benchmark::State& state) {
  Renderer r;
  for (auto _ : state) {
    r.loadFontFace(FONT_PATH);
  }
}
BENCHMARK(LoadFontFace);

static void RenderAtlas(benchmark::State& state) {
  Renderer r;
  r.loadFontFace(FONT_PATH);
  for (auto _ : state) {
    auto [mat, err] = r.renderAtlas();
    benchmark::DoNotOptimize(mat);
  }
}
BENCHMARK(RenderAtlas);

static void RenderAndWriteAtlas(benchmark::State& state) {
  Renderer r;
  r.loadFontFace(FONT_PATH);
  const auto output_dirname = fs::temp_directory_path();
  const std::string output_filename = "foo";
  for (auto _ : state) {
    auto [mat, err] = r.renderAtlas();
    r.saveImage(output_dirname, output_filename, mat,
                static_cast<ImageWriteStyle>(state.range(0)));
  }
}
BENCHMARK(RenderAndWriteAtlas)
    ->DenseRange(0,
                 static_cast<std::underlying_type_t<ImageWriteStyle>>(
                     ImageWriteStyle::COUNT) -
                     1,
                 1);

BENCHMARK_MAIN();
