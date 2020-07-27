cmake_minimum_required(VERSION 3.17)

project(
  FontRendering
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(third_party)

find_package(Freetype REQUIRED)
find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)
find_package(benchmark REQUIRED)
find_library(GLOG_LIBRARIES glog REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(Tesseract tesseract)
find_package(GTest REQUIRED)

add_executable(StandaloneSentencesDemo src/standalone_sentences.cpp)
target_include_directories(StandaloneSentencesDemo PRIVATE
    ${FREETYPE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(StandaloneSentencesDemo PRIVATE
    ${FREETYPE_LIBRARIES}
    ${OpenCV_LIBS}
    fmt::fmt
    ${GFLAGS_LIBRARIES}
)

add_library(Renderer src/renderer.cpp)
target_include_directories(Renderer PUBLIC 
    include
    ${FREETYPE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(Renderer PUBLIC
    ${FREETYPE_LIBRARIES}
    ${OpenCV_LIBS}
    fmt::fmt
)

add_executable(MoveBadFonts src/move_bad_fonts.cpp)
target_include_directories(MoveBadFonts PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(MoveBadFonts PRIVATE
    Renderer
    ${GFLAGS_LIBRARIES}
)

add_executable(IdentifyDuplicates src/identify_duplicates.cpp)
target_include_directories(IdentifyDuplicates PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(IdentifyDuplicates PRIVATE
    Renderer
    ${GFLAGS_LIBRARIES}
)

add_executable(MoveDuplicates src/move_duplicates.cpp)
target_include_directories(MoveDuplicates PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(MoveDuplicates PRIVATE
    Renderer
    ${GFLAGS_LIBRARIES}
)

add_executable(ShowAtlas src/show_atlas.cpp)
target_include_directories(ShowAtlas PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(ShowAtlas PRIVATE
    Renderer
    ${GFLAGS_LIBRARIES}
)

# add_executable(RenderAtlas src/render_atlas.cpp)
# target_include_directories(RenderAtlas PRIVATE ${GFLAGS_INCLUDE_DIRS})
# target_link_libraries(RenderAtlas PRIVATE Renderer ${GFLAGS_LIBRARIES}
# ${GLOG_LIBRARIES})

add_executable(OcrDemo src/ocr_demo.cpp)
target_include_directories(OcrDemo PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
    ${Tesseract_INCLUDE_DIRS}
)
target_link_libraries(OcrDemo PRIVATE
    Renderer
    ${GFLAGS_LIBRARIES}
    ${Tesseract_LIBRARIES}
    Iosifovitch
)

# Things that fall under the test/ folder, rather than in src/
add_executable(RendererBenchmark test/render_bench.cpp)
target_link_libraries(RendererBenchmark PRIVATE
    Renderer
    benchmark::benchmark
)

# enable_testing()
add_executable(RendererTest test/render_test.cpp)
target_include_directories(RendererTest PRIVATE
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(RendererTest PRIVATE
    Renderer
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

add_executable(LevenshteinTest test/levenshtein_test.cpp)
target_include_directories(LevenshteinTest PRIVATE
    include
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(LevenshteinTest PRIVATE
    ${GTEST_BOTH_LIBRARIES}
    pthread
    Iosifovitch
)

add_executable(ImageWriteBenchmark test/image_write.cpp)
target_include_directories(ImageWriteBenchmark PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(ImageWriteBenchmark PRIVATE
    ${OpenCV_LIBS}
    fmt::fmt
    ${GFLAGS_LIBRARIES}
    benchmark::benchmark
)

add_executable(RecursiveFontList test/recursive_font_list.cpp)
target_include_directories(RecursiveFontList PRIVATE
    include
    ${GFLAGS_INCLUDE_DIRS}
)
target_link_libraries(RecursiveFontList PRIVATE
    fmt::fmt
    ${GFLAGS_LIBRARIES}
)

