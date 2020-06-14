#include <benchmark/benchmark.h>

#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fmt/format.h>

#include "font_rendering/renderer.h"
namespace fs = std::filesystem;

std::string FONT_PATH =
    "/data/fun_with_fonts/font_rendering/test_fonts/ttf/0000000.ttf";

static void FT2_Init(benchmark::State& state) {
  for (auto _ : state) {
    Renderer r;
    benchmark::DoNotOptimize(r);
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
BENCHMARK(RenderAndWriteAtlas)->DenseRange(0, 3, 1);

class ImageWriteFixture : public benchmark::Fixture {
 public:
  inline static const std::vector<std::string> filetypes{

      // These options can be configured manually, and so get their own special
      // benchmark functions.
      ".png",  // Can use a variety of compression values, 0 - 9
      // .pbm is binary output (lossy), .ppm is color, and don't have .pxm
      // writer. .pgm is a bunch of bits on disk
      /* ".pbm", */ ".pgm",
      /* ".ppm", */ /* ".pxm", */ ".pnm",
      // Tiff has a bunch of different compression types.
      ".tiff", ".tif",
      
      ".bmp",
      ".dib",   // Just a bunch of raw bits on disk, no configuration
      ".webp",  // Lossless compression only at quality = 100, which is default

      // These formats aren't lossless
      // ".sr", ".ras", ".hdr", ".pic",

      // Jpeg isn't lossless. Technically, there are lossless versions, but
      // probably not supported by libjpeg.
      // ".jpeg", ".jpg", ".jpe",

      // I don't have an EXR writer compiled into my OpenCV build
      // ".exr",
  };

  fs::path output_path;
  std::string output_string;
  std::string current_filetype;
  Renderer r;
  cv::Mat mat;
  bool write = false;

  void SetUp(const ::benchmark::State& state) {
    if (state.range(0) >= filetypes.size()) {
      throw std::runtime_error(fmt::format("Filetype count {} out of bounds {}",
                                           state.range(0), filetypes.size()));
    }
    current_filetype = filetypes[state.range(0)];
    output_path = fs::temp_directory_path() / ("image" + current_filetype);
    output_string = output_path.string();

    r.loadFontFace(FONT_PATH);
    auto ret = r.renderAtlas();
    if (std::get<1>(ret) != RendererError::None) {
      throw std::runtime_error("Failed to render atlas");
    }
    mat = std::move(std::get<0>(ret));
  }

  void TearDown(const ::benchmark::State& state) {
    // Let's check that the compression was in fact lossless
    if (!write) {
      // Maybe do something here
      return;
    }

    auto read = cv::imread(output_string, cv::IMREAD_GRAYSCALE);

    try {
      if (cv::countNonZero(mat != read) != 0) {
        fmt::print("WARNING: Compression was not lossless for file {}!\n", output_string);
      }
    } catch (...) {
      fmt::print("WARNING: Exception when comparing file {}!\n", output_string);
    }
  }
};

BENCHMARK_DEFINE_F(ImageWriteFixture, WriteFiletype)(benchmark::State& state) {
  state.SetLabel(current_filetype);
  for (auto _ : state) {
    write = cv::imwrite(output_string, mat);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WriteFiletype)
    ->DenseRange(5, 7, 1);

BENCHMARK_DEFINE_F(ImageWriteFixture, WritePng)(benchmark::State& state) {
  const std::vector<int> compression_params{
      cv::IMWRITE_PNG_COMPRESSION, static_cast<int>(state.range(1)),
      cv::IMWRITE_PNG_STRATEGY,    static_cast<int>(state.range(2)),
      // Using PNG bilevel that's not 0 (default) makes it not lossless.
      // cv::IMWRITE_PNG_BILEVEL,     static_cast<int>(state.range(3)),
  };
  state.SetLabel(fmt::format("{} {} compression {} strategy",
                             current_filetype, state.range(1), state.range(2)
                             ));

  for (auto _ : state) {
    write = cv::imwrite(output_string, mat, compression_params);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}
static void PngParameters(benchmark::internal::Benchmark* b) {
  const std::vector<int> png_strategies{
      cv::IMWRITE_PNG_STRATEGY_DEFAULT,      cv::IMWRITE_PNG_STRATEGY_FILTERED,
      cv::IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, cv::IMWRITE_PNG_STRATEGY_RLE,
      cv::IMWRITE_PNG_STRATEGY_FIXED,
  };
  for (int compression = 0; compression <= 9; ++compression) {
    for (size_t i = 0; i < png_strategies.size(); ++i) {
      b->Args({0, compression, png_strategies[i]});
    }
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WritePng)->Apply(PngParameters);

BENCHMARK_DEFINE_F(ImageWriteFixture, WritePxm)(benchmark::State& state) {
  const std::vector<int> compression_params{cv::IMWRITE_PXM_BINARY,
                                            static_cast<int>(state.range(1))};
  state.SetLabel(fmt::format("{} {} binary", current_filetype, state.range(1)));

  for (auto _ : state) {
    write = cv::imwrite(output_string, mat, compression_params);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WritePxm)
    ->Args({1, 0})
    ->Args({1, 1})
    ->Args({2, 0})
    ->Args({2, 1});

BENCHMARK_DEFINE_F(ImageWriteFixture, WriteTiff)(benchmark::State& state) {
  const std::vector<int> compression_params{cv::IMWRITE_TIFF_COMPRESSION,
                                            static_cast<int>(state.range(1))};
  state.SetLabel(
      fmt::format("{} {} compression", current_filetype, state.range(1)));

  for (auto _ : state) {
    write = cv::imwrite(output_string, mat, compression_params);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}

static void TiffParameters(benchmark::internal::Benchmark* b) {
  // https://www.awaresystems.be/imaging/tiff/tifftags/compression.html
  // These are all the ones that should work, but only a few do.
  // const std::vector<int> tiff_schemes{1,     2,     3,     4,     5,     6,
  //                                     7,     32766, 32771, 32773, 32809,
  //                                     32895, 32896, 32897, 32898, 32908,
  //                                     32909, 32946, 8,     32947, 34661,
  //                                     34676, 34677, 34712};
  const std::vector<int> tiff_schemes{
      1, 5, 32909, 32946, 8,
  };
  for (const auto scheme : tiff_schemes) {
    b->Args({3, scheme});
    b->Args({4, scheme});
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WriteTiff)->Apply(TiffParameters);

BENCHMARK_MAIN();
