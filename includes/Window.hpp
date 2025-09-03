#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <string>
#include <vulkan/vulkan.h> 
#include <stdexcept>

class Window
{
private:
    int width;
    int height;
    const std::string name;
    bool frameBufferResized = false;

    void initWindow();
    GLFWwindow *display;
    static void  frameBufferResizeCallback(GLFWwindow *window, int width, int height);
public:
    Window(int w, int h, std::string name);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    bool shouldClose() {return glfwWindowShouldClose(display); }
    void createWindowSurface(VkInstance instance, VkSurfaceKHR *surface);
    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};}
    bool wasWindowResized() {return frameBufferResized;}
    void resetWindowResizedFlag() {frameBufferResized = false;}

    int getWidth() { return width; }
    int getHeight() { return height; }

};

