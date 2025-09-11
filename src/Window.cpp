#include "Window.hpp"

Window::Window(int w, int h, std::string name) : width(w), height(h), name(name)
{
    initWindow();
}

Window::~Window()
{
    glfwDestroyWindow(display);
    glfwTerminate();
}


void Window::initWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    display = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(display, this);
    glfwSetFramebufferSizeCallback(display, frameBufferResizeCallback);
}

void  Window::createWindowSurface(VkInstance instance, VkSurfaceKHR *surface)
{
    if(glfwCreateWindowSurface(instance, display, nullptr, surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");

}

void  Window::frameBufferResizeCallback(GLFWwindow *window, int width, int height)
{
    auto nwindow = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    nwindow->frameBufferResized = true;
    nwindow->width = width;
    nwindow->height = height;
}
