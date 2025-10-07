#pragma once

#include "Window.hpp"
#include "Pipeline.hpp"
#include "clay.h"
#include "Device.hpp"
#include "SwapChain.hpp"
#include "Object.hpp"
#include <optional>
#include <set>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <initializer_list>


#define VALIDATE true

struct QueueFamiliyIndies
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value(); }
};


class ShowcaseApp
{
private:
    VkInstance instance;
    Window window;
    VkDevice device;

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkImageView> swapChainImageViews;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> imageRenderFinished;
    std::vector<VkFence>     inFlightFences;
    std::vector<VkFence>     imagesInFlight;

    uint32_t currentFrame = 0;

    VkImage        offscreenImage = VK_NULL_HANDLE;
    VkDeviceMemory offscreenMemory = VK_NULL_HANDLE;
    VkImageView    offscreenView = VK_NULL_HANDLE;

    VkDescriptorSetLayout computeSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      computeDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet       computeSet       = VK_NULL_HANDLE;

    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       computePipeline       = VK_NULL_HANDLE;

    VkRenderPass          renderPass = VK_NULL_HANDLE;

    VkSampler             offscreenSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout graphicsSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      graphicsDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet       graphicsSet       = VK_NULL_HANDLE;

    VkPipelineLayout      graphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            graphicsPipeline       = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    bool offscreenInitialized = false;
    
    static void imageBarrier(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, 
                            VkPipelineStageFlags dstStage, VkAccessFlags dstAccess, VkImageLayout oldLayout, VkImageLayout newLayout);
    
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSurfaceKHR surface;
    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    SBVH bottomLevelAS;
    std::vector<MaterialGPU> materials;
    
    
    std::vector<const char *> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void setupDebugMessenger();
    QueueFamiliyIndies findQueueFamilies(VkPhysicalDevice device);
    void createLogicalDevice();
    void createSwapchain();
    void recreateSwapchain();
    void createImageViews();
    void createSyncObjects();
    void createOffscreenTarget();
    void destroyOffscreenTarget();
    void createComputeDescriptors();
    void updateComputeDescriptor();
    void destroyComputeDescriptors();
    void createComputePipeline();
    void createRenderPass();
    void createFramebuffers();
    void destroyFramebuffers();
    void createGraphicsDescriptors();
    void destroyGraphicsDescriptors();
    void createFullscreenGraphicsPipeline();
    void createCommandPoolAndBuffers();
    void destroyCommandPoolAndBuffers();
    void recordFrameCommands(VkCommandBuffer cmd, uint32_t imageIndex);
    SwapChainSupportDetails querrySwapchaindetails(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    inline VkSwapchainKHR validOldSwapchain() const { return swapChain != VK_NULL_HANDLE ? swapChain : VK_NULL_HANDLE; }

public:
    ShowcaseApp(VkPhysicalDevice gpu, VkInstance inst, SBVH sbvh, std::vector<MaterialGPU> material);
    ~ShowcaseApp();
    void run();
};