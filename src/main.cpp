#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>
#include <array>
#include <cstdint>
#include "UiApp.hpp"

int main(int argc, char *argv[]) {
  std::string def;
  if (argc > 1)
    def = argv[1];
  UiApp app(def);

  try
  {
    app.run();
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  

}