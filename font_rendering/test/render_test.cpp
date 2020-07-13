#include <gtest/gtest.h>

#include "font_rendering/renderer.h"

TEST(TestRendererSpacing, Default) {
  RendererSpacing stats;
  EXPECT_EQ(stats.dpi, 110);
  EXPECT_EQ(stats.point, 72);
  EXPECT_EQ(stats.em, 110);
  EXPECT_EQ(stats.half_em, 55);
  EXPECT_EQ(stats.atlas_border, 13);
  EXPECT_EQ(stats.atlas_padding, 55);
}

TEST(TestRendererSpacing, OnlyDpi) {
  RendererSpacing stats(110);
  EXPECT_EQ(stats.dpi, 110);
  EXPECT_EQ(stats.point, 72);
  EXPECT_EQ(stats.em, 110);
  EXPECT_EQ(stats.half_em, 55);
  EXPECT_EQ(stats.atlas_border, 13);
  EXPECT_EQ(stats.atlas_padding, 55);
}

TEST(TestRendererSpacing, HighDpiAndPt) {
  RendererSpacing stats(300, 12);
  EXPECT_EQ(stats.dpi, 300);
  EXPECT_EQ(stats.point, 12);
  EXPECT_EQ(stats.em, 50);
  EXPECT_EQ(stats.half_em, 25);
  EXPECT_EQ(stats.atlas_border, 6);
  EXPECT_EQ(stats.atlas_padding, 25);
}

TEST(TestRendererSpacing, AllParameters) {
  RendererSpacing stats(300, 12, 30, 20);
  EXPECT_EQ(stats.dpi, 300);
  EXPECT_EQ(stats.point, 12);
  EXPECT_EQ(stats.em, 50);
  EXPECT_EQ(stats.half_em, 25);
  EXPECT_EQ(stats.atlas_border, 30);
  EXPECT_EQ(stats.atlas_padding, 20);
}

TEST(TestRendererSpacing, Initializer) {
  RendererSpacing stats{300, 12, 30, 20};
  EXPECT_EQ(stats.dpi, 300);
  EXPECT_EQ(stats.point, 12);
  EXPECT_EQ(stats.em, 50);
  EXPECT_EQ(stats.half_em, 25);
  EXPECT_EQ(stats.atlas_border, 30);
  EXPECT_EQ(stats.atlas_padding, 20);
}

TEST(TestRendererSpacing, BadValues) {
  RendererSpacing stats(-10, -10);
  EXPECT_EQ(stats.dpi, 110);
  EXPECT_EQ(stats.point, 72);
  EXPECT_EQ(stats.em, 110);
  EXPECT_EQ(stats.half_em, 55);
  EXPECT_EQ(stats.atlas_border, 13);
  EXPECT_EQ(stats.atlas_padding, 55);
}
