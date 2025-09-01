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

int main() {
  UiApp app;

  try
  {
    app.run();
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  

}