cmake_minimum_required(VERSION 3.17)

project(
  FontRendering
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Freetype REQUIRED)
find_package(OpenCV COMPONENTS core imgproc highgui REQUIRED)
find_package(fmt REQUIRED)
find_package(gflags REQUIRED)
find_package(benchmark REQUIRED)
find_library(GLOG_LIBRARIES glog REQUIRED)

add_executable(StandaloneSentencesDemo src/standalone_sentences.cpp)
target_include_directories(
  StandaloneSentencesDemo PRIVATE ${FREETYPE_INCLUDE_DIRS}
                                  ${OpenCV_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIRS})
target_link_libraries(
  StandaloneSentencesDemo PRIVATE ${FREETYPE_LIBRARIES} ${OpenCV_LIBS} fmt::fmt
                                  ${GFLAGS_LIBRARIES})

add_library(Renderer src/renderer.cpp)
target_include_directories(Renderer PUBLIC include ${FREETYPE_INCLUDE_DIRS}
                                           ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Renderer PUBLIC ${FREETYPE_LIBRARIES} ${OpenCV_LIBS}
                                      fmt::fmt)

add_executable(ShowAtlas src/show_atlas.cpp)
target_include_directories(ShowAtlas PRIVATE ${GFLAGS_INCLUDE_DIRS})
target_link_libraries(ShowAtlas PRIVATE Renderer ${GFLAGS_LIBRARIES})

add_executable(RenderAtlas src/render_atlas.cpp)
target_include_directories(RenderAtlas PRIVATE ${GFLAGS_INCLUDE_DIRS})
target_link_libraries(RenderAtlas PRIVATE Renderer ${GFLAGS_LIBRARIES}
                                          ${GLOG_LIBRARIES})

add_executable(RendererBenchmark test/render_bench.cpp)
target_link_libraries(RendererBenchmark PRIVATE Renderer benchmark::benchmark)

add_executable(ImageWriteBenchmark test/image_write.cpp)
target_include_directories(ImageWriteBenchmark PRIVATE ${OpenCV_INCLUDE_DIRS}
                                                       ${GFLAGS_INCLUDE_DIRS})
target_link_libraries(
  ImageWriteBenchmark PRIVATE ${OpenCV_LIBS} fmt::fmt ${GFLAGS_LIBRARIES}
                              benchmark::benchmark)
