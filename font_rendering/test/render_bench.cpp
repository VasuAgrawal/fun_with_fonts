#include "font_rendering/renderer.h"

#include <benchmark/benchmark.h>

static void BM_FT2_Init(benchmark::State& state) {
  for (auto _ : state) {
    Renderer r;
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_FT2_Init);

static void BM_LoadFontFace(benchmark::State& state) {
  Renderer r;
  for (auto _ : state) {
    r.loadFontFace("/data/fun_with_fonts/font_rendering/test_fonts/ttf/0000000.ttf");
  }
}
BENCHMARK(BM_LoadFontFace);


static void BM_RenderAtlas(benchmark::State& state) {
  Renderer r;
  r.loadFontFace("/data/fun_with_fonts/font_rendering/test_fonts/ttf/0000000.ttf");
  for (auto _ : state) {
    auto [mat, err] = r.renderAtlas();
    benchmark::DoNotOptimize(mat);
  }
}
BENCHMARK(BM_RenderAtlas);

BENCHMARK_MAIN();
