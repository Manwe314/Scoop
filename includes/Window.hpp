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

    VkExtent2D getExtent() { return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};}
    GLFWwindow* handle() const { return display; }
    
    bool shouldClose() {return glfwWindowShouldClose(display); }
    bool wasWindowResized() {return frameBufferResized;}
    bool isMouseDown(int button) const { return glfwGetMouseButton(display, button) == GLFW_PRESS; }
    
    void createWindowSurface(VkInstance instance, VkSurfaceKHR *surface);
    void resetWindowResizedFlag() {frameBufferResized = false;}
    void getCursorPos(double& x, double& y) const { glfwGetCursorPos(display, &x, &y); }
    void getSizes(int& winW, int& winH, int& fbW, int& fbH) const { glfwGetWindowSize(display, &winW, &winH); glfwGetFramebufferSize(display, &fbW, &fbH); }
    void close() { glfwSetWindowShouldClose(display, GLFW_TRUE); }
    void recreate();
    
    int getWidth() { return width; }
    int getHeight() { return height; }

};

