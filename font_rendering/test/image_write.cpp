

#include <benchmark/benchmark.h>
#include <fmt/format.h>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <gflags/gflags.h>

namespace fs = std::filesystem;

DEFINE_string(image_path, "", "Path to image to benchmark against");
DEFINE_bool(rgb_image, false, "Whether the image should be loaded as RGB");

class ImageWriteFixture : public benchmark::Fixture {
 public:
  inline static const std::vector<std::string> filetypes{
    // Trying to deal with all of the supported filetypes listed here, or at
    // least as many as are reasonable:
    // https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56

      // These options can be configured manually, and so get their own special
      // benchmark functions.
      
      // .pbm is binary output (lossy), .ppm is color, and don't have .pxm
      // writer. .pgm is a bunch of bits on disk
      ".pbm", ".pgm",
      ".ppm", /* ".pxm", */ ".pnm",

      // Can use a variety of compression values, 0 - 9
      ".png",  
      
      // Tiff has a bunch of different compression types.
      ".tiff", /* ".tif", */

      // Lossless compression only at quality = 100, which is default
      ".webp",  
      
      // Jpeg isn't lossless. Technically, there are lossless versions, but
      // probably not supported by libjpeg.
      /* ".jpeg", */ ".jpg", /* ".jpe", */
      
      // I don't have an EXR writer compiled into my OpenCV build
      // ".exr",

      // The rest of these don't offer any configuration through opencv, so they
      // get written with the generic "WriteFiletype" function.
      ".bmp", ".dib",   

      // These formats aren't lossless
      ".sr", ".ras", /*".hdr", ".pic", */
  };

  fs::path output_path;
  std::string output_string;
  std::string current_filetype;
  // Renderer r;
  cv::Mat mat;
  bool write = false;

  void SetUp(const ::benchmark::State& state) {
    write = false;

    if (state.range(0) >= filetypes.size()) {
      throw std::runtime_error(fmt::format("Filetype count {} out of bounds {}",
                                           state.range(0), filetypes.size()));
    }
    current_filetype = filetypes[state.range(0)];
    output_path = fs::temp_directory_path() / ("image" + current_filetype);
    output_string = output_path.string();
    
    mat = cv::imread(FLAGS_image_path,
        FLAGS_rgb_image ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);

    // Store size in the benchmark state for easy comparison
    auto& mutable_state = const_cast<::benchmark::State&>(state);
    mutable_state.counters["bytes_size"] = mat.total() * mat.elemSize();
    mutable_state.counters["pixel_count"] = mat.total();
    // r.loadFontFace(FONT_PATH);
    // auto ret = r.renderAtlas();
    // if (std::get<1>(ret) != RendererError::None) {
    //   throw std::runtime_error("Failed to render atlas");
    // }
    // mat = std::move(std::get<0>(ret));
  }

  void TearDown(const ::benchmark::State& state) {
    // Let's check that the compression was in fact lossless
    if (!write) {
      // Maybe do something here
      return;
    }

    auto read = cv::imread(output_string,
        FLAGS_rgb_image ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);

    uint64_t nonzero = 0;
    try {
      const cv::Mat comp = mat != read;

      if (FLAGS_rgb_image) {
        for (auto it = comp.begin<cv::Vec3b>(), end = comp.end<cv::Vec3b>(); it != end; ++it) {
          nonzero += (
              static_cast<cv::Vec3b>(*it)[0] || 
              static_cast<cv::Vec3b>(*it)[1] || 
              static_cast<cv::Vec3b>(*it)[2]);
          // nonzero += *it[0] || *it[1] || *it[2];
        // nonzero += 3;
      }
      } else {
        for (auto it = comp.begin<uint8_t>(), end = comp.end<uint8_t>(); it != end; ++it) {
          nonzero += static_cast<bool>(*it);
          // ++nonzero;
          // const uint8_t* p = comp.ptr<uint8_t>(i);
        } 

      }



      // nonzero = cv::sum(mat != read);
      // (mat != read).forEach<uint8_t>([](uint8_t& p, auto _)
      
      // nonzero = cv::countNonZero(mat != read);
      // if (nonzero = cv::countNonZero(mat != read)) {
      //   fmt::print("WARNING: Compression was not lossless for file {}!\n",
      //              output_string);
      // }
    } catch (...) {
      fmt::print(stderr, "WARNING: Exception when comparing file {}!\n", output_string);
    }

    auto& mutable_state = const_cast<::benchmark::State&>(state);
    mutable_state.counters["changed_pixels"] = nonzero;
  }
};

BENCHMARK_DEFINE_F(ImageWriteFixture, WritePxm)(benchmark::State& state) {
  // ppm doesn't support rgb, so special case on that.
  if (current_filetype  == ".ppm" && !FLAGS_rgb_image) {
    state.SkipWithError(".ppm doesn't support grayscale");
    return;
  } else if ((current_filetype == ".pgm" || current_filetype == ".pbm") && FLAGS_rgb_image) {
    state.SkipWithError(fmt::format("{} doesn't support RGB images", current_filetype).c_str());
    return;
  }

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
static void PxmParameters(benchmark::internal::Benchmark* b) {
  for (int i = 0; i <= 3; ++i) { // Range pointing to pxm filetypes
    for (int binary = 0; binary <= 1; ++binary) { // Binary output or not
      b->Args({i, binary});
    }
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WritePxm)
  ->Apply(PxmParameters);



BENCHMARK_DEFINE_F(ImageWriteFixture, WritePng)(benchmark::State& state) {
  const std::vector<int> compression_params{
      cv::IMWRITE_PNG_COMPRESSION, static_cast<int>(state.range(1)),
      cv::IMWRITE_PNG_STRATEGY, static_cast<int>(state.range(2)),
      // Using PNG bilevel that's not 0 (default) makes it not lossless.
      // cv::IMWRITE_PNG_BILEVEL,     static_cast<int>(state.range(3)),
  };
  state.SetLabel(fmt::format("{} {} compression {} strategy", current_filetype,
                             state.range(1), state.range(2)));

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
      b->Args({4 /* .png index in filetypes */, compression, png_strategies[i]});
    }
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WritePng)->Apply(PngParameters);





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
    b->Args({5 /* .tiff index in filetypes */, scheme});
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WriteTiff)->Apply(TiffParameters);





BENCHMARK_DEFINE_F(ImageWriteFixture, WriteWebp)(benchmark::State& state) {
  const std::vector<int> compression_params{cv::IMWRITE_WEBP_QUALITY,
                                            static_cast<int>(state.range(1))};
  state.SetLabel(
      fmt::format("{} {} quality", current_filetype, state.range(1)));

  for (auto _ : state) {
    write = cv::imwrite(output_string, mat, compression_params);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}
static void WebpParameters(benchmark::internal::Benchmark* b) {
  // Somewhat arbitrarily chosen quality values, notably including 101 which is
  // necessarily lossless.
  for (int quality = 1; quality <= 101; quality += 10) {
    b->Args({6 /* .webp index in filetypes */, quality});
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WriteWebp)->Apply(WebpParameters);



BENCHMARK_DEFINE_F(ImageWriteFixture, WriteJpeg)(benchmark::State& state) {
  const std::vector<int> compression_params{
    cv::IMWRITE_JPEG_QUALITY, static_cast<int>(state.range(1)),
    cv::IMWRITE_JPEG_PROGRESSIVE, static_cast<int>(state.range(2)),
    cv::IMWRITE_JPEG_OPTIMIZE, static_cast<int>(state.range(3))
  };
  state.SetLabel(
      fmt::format("{} {} quality {} progressive {} optimize", 
        current_filetype, state.range(1), state.range(2), state.range(3)));

  for (auto _ : state) {
    write = cv::imwrite(output_string, mat, compression_params);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}
static void JpegParameters(benchmark::internal::Benchmark* b) {
  std::vector<int> qualities { 95 /* start with default */ };
  for (int q = 0; q <= 100; q += 20) {
    // And add a smattering of others
    qualities.push_back(q);
  }

  for (const auto quality : qualities) {
    for (int progressive = 0; progressive <= 1; ++progressive) {
      for (int optimize = 0; optimize <= 1; ++optimize) {
        b->Args({7 /* .jpeg index in filetypes */, quality, progressive, optimize});
      }
    }
  }
}
BENCHMARK_REGISTER_F(ImageWriteFixture, WriteJpeg)->Apply(JpegParameters);













BENCHMARK_DEFINE_F(ImageWriteFixture, WriteFiletype)(benchmark::State& state) {
  state.SetLabel(current_filetype);
  for (auto _ : state) {
    write = cv::imwrite(output_string, mat);
  }

  if (write) {
    state.counters["size"] = fs::file_size(output_path);
  }
}
// Dense range corresponds to the filetypes that aren't handled by one of the
// more specialized methods above (their indices in the vector).
BENCHMARK_REGISTER_F(ImageWriteFixture, WriteFiletype)->DenseRange(8, 11, 1);


int main(int argc, char* argv[]) {
  gflags::AllowCommandLineReparsing();
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
