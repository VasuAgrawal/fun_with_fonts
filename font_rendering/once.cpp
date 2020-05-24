#include "renderer.h"

int main(int argc, char* argv[]) {
  std::string loc = "/zfs/home/vasu/Documents/projects/fun_with_fonts/sources/d3mok/static.d3mok.net/fonts/equity/EquityTextA.otf";
  Renderer r;
  r.loadFontFace(loc);
  for (int i = 0; i < 100; ++i) {
    auto mat = r.renderAtlas();
  }

  return 0;
}
