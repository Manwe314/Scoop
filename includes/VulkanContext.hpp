#pragma once

#include <vulkan/vulkan.h> 
#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include "Window.hpp"

#ifdef NDEBUG
  const bool enableValidationLayers = false;
#else
  const bool enableValidationLayers = true;
#endif

class VulkanContext
{
private:
    VkInstance instance;
    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

    void createInstance();
    bool checkValidationLayerSupport();
    std::vector<const char *> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void hasGflwRequiredInstanceExtensions();
public:
    VulkanContext();
    ~VulkanContext();
    VkInstance getInstance() { return instance; }
};

