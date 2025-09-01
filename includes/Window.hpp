#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <vulkan/vulkan.h> 
#include <stdexcept>

class Window
{
private:
    const int width;
    const int height;
    const std::string name;

    void initWindow();
    GLFWwindow *display;
public:
    Window(int w, int h, std::string name);
    ~Window();
    Window(const Window&) = delete;
    void operator=(const Window&) = delete;
    bool shouldClose() {return glfwWindowShouldClose(display); };
    void createWindowSurface(VkInstance instance, VkSurfaceKHR *surface);
    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};};
};

