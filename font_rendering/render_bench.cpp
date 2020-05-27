#include "renderer.h"

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
    r.loadFontFace("/zfs/home/vasu/Documents/projects/fun_with_fonts/ttf/pepperlandexpand.ttf");
  }
}
BENCHMARK(BM_LoadFontFace);


static void BM_RenderAtlas(benchmark::State& state) {
  Renderer r;
  r.loadFontFace("/zfs/home/vasu/Documents/projects/fun_with_fonts/ttf/pepperlandexpand.ttf");
  for (auto _ : state) {
    benchmark::DoNotOptimize(r.renderAtlas());
  }
}
BENCHMARK(BM_RenderAtlas);

BENCHMARK_MAIN();
