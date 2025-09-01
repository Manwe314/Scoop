#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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


UiApp::UiApp() : window(WIDTH, HEIGHT, "UI"),\
                 device(window),
                 swapChain(device, window.getExtent())
{
    loadModels();
    createPipelineLayout();
    createPipeline();
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
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    if (vkCreatePipelineLayout(device.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout");
}

void UiApp::createPipeline()
{
    auto pipelineConfig = Pipeline::defaultPipelineConfigIngo(swapChain.width(), swapChain.height());
    pipelineConfig.renderPass = swapChain.getRenderPass();
    pipelineConfig.pipelineLayout = pipelineLayout;
    pipeline = std::make_unique<Pipeline>(device,  pipelineConfig, "build/shaders/rect.vert.spv", "build/shaders/rect.frag.spv");

}

void UiApp::createCommandBuffers()
{
    commandBuffers.resize(swapChain.imageCount());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = device.getCommandPool();
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device.device(), &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");
    
    
    for (int i = 0; i < commandBuffers.size(); i++)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording command buffer!");

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.renderPass = swapChain.getRenderPass();
        renderPassInfo.framebuffer = swapChain.getFrameBuffer(i);

        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChain.getSwapChainExtent();

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {0.3f, 0.3f, 0.3f, 1.0f};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount =  static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        pipeline->bind(commandBuffers[i]);
        model->bind(commandBuffers[i]);
        model->draw(commandBuffers[i]);

        vkCmdEndRenderPass(commandBuffers[i]);
        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to record Command buffer!");
    }
}

void UiApp::drawFrame()
{
    uint32_t imageIndex;
    auto result = swapChain.acquireNextImage(&imageIndex);

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to aquire swap chain image");

    result = swapChain.submitCommandBuffers(&commandBuffers[imageIndex], &imageIndex);
    if (result != VK_SUCCESS)
        throw std::runtime_error("Failed to present swap chain image");
}

void UiApp::loadModels()
{
    std::vector<Model::Vertex> vertices {
        {{0.0f, -0.5f}},
        {{0.5f, 0.5f}},
        {{-0.5f, 0.5f}}
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