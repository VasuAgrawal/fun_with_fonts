add_library(Iosifovitch src/levenshtein.cpp)
target_include_directories(
    Iosifovitch PRIVATE
    include/iosifovitch
    PUBLIC include
)
