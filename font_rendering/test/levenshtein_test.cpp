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
