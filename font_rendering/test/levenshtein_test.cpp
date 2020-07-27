#include "font_rendering/levenshtein.h"

#include <gtest/gtest.h>

#include "iosifovitch/iosifovitch.hpp"

TEST(TestNaive, Simple) {
  EXPECT_EQ(levenshteinNaive("test", "test"), 0);
  EXPECT_EQ(levenshteinNaive("test", "tent"), 1);
  EXPECT_EQ(levenshteinNaive("gumbo", "gambol"), 2);
  EXPECT_EQ(levenshteinNaive("kitten", "sitting"), 3);
  EXPECT_EQ(levenshteinNaive("Saturday", "Sunday"), 3);
  EXPECT_EQ(levenshteinNaive("the quick brown fox \njumps over the lazy dog",
                             "the quick brown fox \njumps over the lazy dog"),
            0);
}

TEST(TestIosifovitch, Simple) {
  EXPECT_EQ(iosifovitch::levenshtein_distance("test", "test"), 0);
  EXPECT_EQ(iosifovitch::levenshtein_distance("test", "tent"), 1);
  EXPECT_EQ(iosifovitch::levenshtein_distance("gumbo", "gambol"), 2);
  EXPECT_EQ(iosifovitch::levenshtein_distance("kitten", "sitting"), 3);
  EXPECT_EQ(iosifovitch::levenshtein_distance("Saturday", "Sunday"), 3);
  EXPECT_EQ(iosifovitch::levenshtein_distance(
                "the quick brown fox \njumps over the lazy dog",
                "the quick brown fox \njumps over the lazy dog"),
            0);
}

// The reinterpret_cast is necessary because we end up with a const char8_t[]
// for the utf8 string, which isn't what we want. However, it can be safely
// reinterpretd based on this: https://stackoverflow.com/a/57453713/893643
//
// See also this helpful link from microsoft:
// https://docs.microsoft.com/en-us/cpp/cpp/string-and-character-literals-cpp?view=vs-2019

TEST(Utf8AdapterTest, Sizes) {
  std::string s;

  s = reinterpret_cast<const char*>(u8"f");
  EXPECT_EQ((*Utf8Adapter(s).begin()).size(), 1);

  s = reinterpret_cast<const char*>(u8"܀");
  EXPECT_EQ((*Utf8Adapter(s).begin()).size(), 2);

  s = reinterpret_cast<const char*>(u8"─");
  EXPECT_EQ((*Utf8Adapter(s).begin()).size(), 3);

  s = reinterpret_cast<const char*>(u8"🂡");
  EXPECT_EQ((*Utf8Adapter(s).begin()).size(), 4);
}

TEST(Utf8AdapterTest, AsciiLength) {
  const std::string ascii = reinterpret_cast<const char*>(u8"Hello World");
  size_t count = 0;
  for (const auto& c : Utf8Adapter(ascii)) {
    ++count;
    EXPECT_EQ(c.size(), 1);
  }

  EXPECT_EQ(count, 11);
}

TEST(Utf8AdapterTest, Utf8Length) {
  const std::string utf8 = reinterpret_cast<const char*>(u8"Привет, товарищ");
  size_t count = 0;
  for (const auto& c : Utf8Adapter(utf8)) {
    ++count;
  }

  EXPECT_EQ(count, 15);
}

TEST(ToLower, ToLowerAscii) {
  const std::string input = "HeLlO, w0rLd!";
  const std::string output = "hello, w0rld!";
  EXPECT_EQ(lowerAsciiInUtf8(input), output);
}

TEST(ToLower, ToLowerUtf8) {
  const std::string input = reinterpret_cast<const char*>(u8"🂡helLO, товарищ");
  const std::string output = reinterpret_cast<const char*>(u8"🂡hello, товарищ");
  EXPECT_EQ(lowerAsciiInUtf8(input), output);
}
