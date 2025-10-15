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
#include "ShowcaseApp.hpp"
#include "VulkanContext.hpp"
#include <cstdlib>



int main(int argc, char *argv[]) {
  std::string def;
  if (argc > 1)
    def = argv[1];
  AppState state;
  
  VulkanContext context;
  
  UiApp uiApp(def, context);

  try
  {
    while (true)
    {
      state = uiApp.run();
      if (state.shouldClose == false && state.device == VK_NULL_HANDLE)
        throw std::runtime_error("Ui App Returned a NULL HANDLE");
      else if (state.shouldClose == false && state.device != VK_NULL_HANDLE)
      {
        ShowcaseApp showcaseApp(state.device, context.getInstance(), state.scene);
        showcaseApp.run();
      }
      if (state.shouldClose == true)
        break;
    }
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  
}