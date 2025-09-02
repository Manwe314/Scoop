#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>


#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <array>

#define CLAY_IMPLEMENTATION
#include "clay.h"

# include "UiApp.hpp"

// static inline void ClayHandleErrors(Clay_ErrorData e) {
//     fprintf(stderr, "[Clay] %.*s\n", (int)e.errorText.length, e.errorText.chars);
// }

// static inline Clay_Dimensions MeasureTextMono(Clay_StringSlice text, Clay_TextElementConfig* cfg, void* user) {
//     return { (float)text.length * cfg->fontSize, (float)cfg->fontSize };
// }


struct SimplePushConstantData {
    glm::vec2 offset;
    alignas(16) glm::vec3 color;
};

UiApp::UiApp() : window(WIDTH, HEIGHT, "UI"), device(window)
{
    loadModels();
    createPipelineLayout();
    recreateSwapchain();
    createCommandBuffers();
    // clayMemSize = Clay_MinMemorySize();
    // clayMem = std::malloc(clayMemSize);
    // if (!clayMem) throw std::runtime_error("Failed to alloc Clay memory");
    // clayArena   = Clay_CreateArenaWithCapacityAndMemory(clayMemSize, clayMem);

    // int w = WIDTH, h = HEIGHT;
    // Clay_Initialize(clayArena, { (float)w, (float)h }, { ClayHandleErrors });
    // Clay_SetMeasureTextFunction(MeasureTextMono, 0);
}

void UiApp::createPipelineLayout()
{

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SimplePushConstantData);


    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout");
}

void UiApp::createPipeline()
{
    assert(swapChain != nullptr && "Cannot create pipeline before the swap chain");
    assert(pipelineLayout != nullptr && "Cannot create pipeline before pipeline Layout");

    PipelineConfigInfo pipelineConfig{};
    Pipeline::defaultPipelineConfigIngo(pipelineConfig);
    pipelineConfig.renderPass = swapChain->getRenderPass();
    pipelineConfig.pipelineLayout = pipelineLayout;
    pipeline = std::make_unique<Pipeline>(device,  pipelineConfig, "build/shaders/rect.vert.spv", "build/shaders/rect.frag.spv");

}

void UiApp::recreateSwapchain()
{
    auto extent = window.getExtent();
    while (extent.width == 0 || extent.height == 0)
    {
        extent = window.getExtent();
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device.device());
    if(swapChain == nullptr)
        swapChain = std::make_unique<SwapChain>(device, extent);
    else {
        swapChain = std::make_unique<SwapChain>(device, extent, std::move(swapChain));
        if (swapChain->imageCount() != commandBuffers.size()){
            freeCommandBuffers();
            createCommandBuffers();
        }
    }
    createPipeline();
}

void UiApp::createCommandBuffers()
{
    commandBuffers.resize(swapChain->imageCount());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device.device(), &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");
}

void UiApp::freeCommandBuffers()
{
    vkFreeCommandBuffers(device.device(), device.getCommandPool(), static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
    commandBuffers.clear();

}

void UiApp::recordCommandBuffer(int imageIndex)
{
    static int frame = 0;
    frame = (frame + 1) % 1000;
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin recording command buffer!");

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.renderPass = swapChain->getRenderPass();
    renderPassInfo.framebuffer = swapChain->getFrameBuffer(imageIndex);
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChain->getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};

    clearValues[0].color = {0.12f, 0.12f, 0.12f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};

    renderPassInfo.clearValueCount =  static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    
    vkCmdBeginRenderPass(commandBuffers[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChain->getSwapChainExtent().width);
    viewport.height = static_cast<float>(swapChain->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 0.0f;
    VkRect2D scissor {{0 , 0}, swapChain->getSwapChainExtent()};
    vkCmdSetViewport(commandBuffers[imageIndex], 0, 1, &viewport);
    vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &scissor);

    pipeline->bind(commandBuffers[imageIndex]);
    model->bind(commandBuffers[imageIndex]);

    for (int j = 0; j < 4; j++)
    {
        SimplePushConstantData push{};
        push.offset = {-0.5f + frame * 0.002f , -0.4f + j * 0.25f};
        push.color = {0.0f, 0.0f, 0.2f + 0.2f *j};

        vkCmdPushConstants(commandBuffers[imageIndex], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SimplePushConstantData), &push);
        model->draw(commandBuffers[imageIndex]);
    }
    

    vkCmdEndRenderPass(commandBuffers[imageIndex]);
    if (vkEndCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS)
        throw std::runtime_error("Failed to record Command buffer!");

}

void UiApp::drawFrame()
{
    uint32_t imageIndex;
    auto result = swapChain->acquireNextImage(&imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
        return;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to aquire swap chain image");

    recordCommandBuffer(imageIndex);
    result = swapChain->submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window.wasWindowResized())
    {
        window.resetWindowResizedFlag();
        recreateSwapchain();
        return;
    }
    if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to present swap chain image");
}

void UiApp::loadModels()
{
    std::vector<Model::Vertex> vertices {
        {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
    };
    model = std::make_unique<Model>(device, vertices);
}

UiApp::~UiApp()
{
    vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
    // std::free(clayMem);
}

// void UiApp::buildUi(float dt, int screenW, int screenH) 
// {

// }

void UiApp::run() {
    while (!window.shouldClose()) 
    {
        glfwPollEvents();
        drawFrame();
    }

    vkDeviceWaitIdle(device.device());
    
}