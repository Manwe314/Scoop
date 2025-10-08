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

struct alignas(16) ParamsGPU {
    glm::mat4 viewProjInv;
    glm::vec4 camPos_time;
    glm::uvec2 imageSize;
    uint32_t   rootIndex;
    uint32_t   _pad0;
};

void copyBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size);


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

    VkBuffer sbvhNodesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory sbvhNodesMemory = VK_NULL_HANDLE;

    VkBuffer triangleBuffer = VK_NULL_HANDLE;
    VkDeviceMemory triangleMemory = VK_NULL_HANDLE;

    VkBuffer shadingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory shadingMemory = VK_NULL_HANDLE;

    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceMemory materialMemory = VK_NULL_HANDLE;

    VkBuffer       paramsBuffer[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory paramsMemory[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    void*          paramsMapped[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

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
    void destorySSBOdata();
    void uploadStaticData();
    void createParamsBuffers();
    void writeStaticComputeBindings();
    void writeParamsBindingForFrame(uint32_t frameIndex);
    void recordFrameCommands(VkCommandBuffer cmd, uint32_t imageIndex);
    void createBuffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags, VkBuffer& outBuf, VkDeviceMemory& outMem);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props, VkPhysicalDevice phys);
    SwapChainSupportDetails querrySwapchaindetails(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    inline VkSwapchainKHR validOldSwapchain() const { return swapChain != VK_NULL_HANDLE ? swapChain : VK_NULL_HANDLE; }

    inline ParamsGPU makeDefaultParams(uint32_t width = 1, uint32_t height = 1, uint32_t rootIndex = 0, float time = 0.0f, const glm::vec3& camPos = glm::vec3(0.0f))
    {
        ParamsGPU p{};
        p.viewProjInv = glm::mat4(1.0f);
        p.camPos_time = glm::vec4(camPos, time);
        p.imageSize   = glm::uvec2(width, height);
        p.rootIndex   = rootIndex;
        p._pad0       = 0;
        return p;
    }

    inline ParamsGPU makeDefaultParams(VkExtent2D extent, uint32_t rootIndex = 0, float time = 0.0f, const glm::vec3& camPos = glm::vec3(0.0f)) { return makeDefaultParams(extent.width, extent.height, rootIndex, time, camPos); }

public:
    ShowcaseApp(VkPhysicalDevice gpu, VkInstance inst, SBVH sbvh, std::vector<MaterialGPU> material);
    ~ShowcaseApp();
    void run();
    template<class T>
    void uploadDeviceLocal(const std::vector<T>& src, VkBufferUsageFlags extraUsage, VkBuffer& outBuf, VkDeviceMemory& outMem)
    {
        VkDeviceSize size = sizeof(T) * src.size();

        VkBuffer stageBuf = VK_NULL_HANDLE;
        VkDeviceMemory stageMem = VK_NULL_HANDLE;
        this->createBuffer(device, physicalDevice, size,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stageBuf, stageMem);

        void* p = nullptr;
        vkMapMemory(device, stageMem, 0, size, 0, &p);
        std::memcpy(p, src.data(), size);
        vkUnmapMemory(device, stageMem);

        this->createBuffer(device, physicalDevice, size,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outBuf, outMem);

        copyBuffer(device, commandPool, graphicsQueue, stageBuf, outBuf, size);

        vkDestroyBuffer(device, stageBuf, nullptr);
        vkFreeMemory(device, stageMem, nullptr);
    }
};

