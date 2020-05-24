#include "renderer.h"

#include <benchmark/benchmark.h>

static void BM_FT2_Init(benchmark::State& state) {
  for (auto _ : state) {
    Renderer r;
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
    r.renderAtlas();
  }
}
BENCHMARK(BM_RenderAtlas);

static void BM_RenderTestText(benchmark::State& state) {
  Renderer r;
  r.loadFontFace("/zfs/home/vasu/Documents/projects/fun_with_fonts/ttf/pepperlandexpand.ttf");
  for (auto _ : state) {
    r.renderTestText();
  }
}
BENCHMARK(BM_RenderTestText);


BENCHMARK_MAIN();
