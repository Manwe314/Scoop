#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <array>

#define CLAY_IMPLEMENTATION
#include "clay.h"
#include "clay_printer.h"

#include "UiApp.hpp"

static inline void ClayHandleErrors(Clay_ErrorData e) {
    fprintf(stderr, "[Clay] %.*s\n", (int)e.errorText.length, e.errorText.chars);
}

static inline Clay_Dimensions MeasureTextMono(Clay_StringSlice text, Clay_TextElementConfig* cfg, void* user) {
    return { (float)text.length * cfg->fontSize, (float)cfg->fontSize };
}


struct SimplePushConstantData {
    glm::mat4 uProj;
};

struct RectangleItem {
    UiRenderer::RectangleInstance instance;
    int zIndex;
    uint32_t seq;
};

UiApp::UiApp() : window(WIDTH, HEIGHT, "UI"), device(window)
{
    loadUi();
    createPipelineLayout();
    recreateSwapchain();
    createCommandBuffers();
    clayMemSize = Clay_MinMemorySize();
    clayMem = std::malloc(clayMemSize);
    if (!clayMem) throw std::runtime_error("Failed to alloc Clay memory");
    clayArena   = Clay_CreateArenaWithCapacityAndMemory(clayMemSize, clayMem);

    int w = WIDTH, h = HEIGHT;
    Clay_Initialize(clayArena, { (float)w, (float)h }, { ClayHandleErrors });
    Clay_SetMeasureTextFunction(MeasureTextMono, 0);
}

void UiApp::createPipelineLayout()
{

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
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
    //change default settings to accomidate clay rendering
    pipelineConfig.inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    pipelineConfig.depthStencilInfo.depthTestEnable = VK_FALSE;
    pipelineConfig.depthStencilInfo.depthWriteEnable = VK_FALSE;
    pipelineConfig.colorBlendAttachment.blendEnable = VK_TRUE;
    pipelineConfig.colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    pipelineConfig.colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineConfig.colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineConfig.colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    pipelineConfig.colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineConfig.colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
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
    buildUi();
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkResetCommandBuffer(commandBuffers[imageIndex], 0);
    if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("Failed to begin recording command buffer!");

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = swapChain->getRenderPass();
    renderPassInfo.framebuffer = swapChain->getFrameBuffer(imageIndex);
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChain->getSwapChainExtent();

    std::array<VkClearValue, 2> clearValues{};

    clearValues[0].color = {0.05f, 0.05f, 0.05f, 1.0f};
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
    viewport.maxDepth = 1.0f;
    VkRect2D scissor {{0 , 0}, swapChain->getSwapChainExtent()};
    vkCmdSetViewport(commandBuffers[imageIndex], 0, 1, &viewport);
    vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &scissor);

    pipeline->bind(commandBuffers[imageIndex]);
    // std::printf("rects this frame: %zu\n", rects.size());
    ui->updateInstances(rects);
    ui->bind(commandBuffers[imageIndex]);

    SimplePushConstantData push_constant{};
    push_constant.uProj = glm::ortho(
        0.0f, (float)swapChain->getSwapChainExtent().width,
        (float)swapChain->getSwapChainExtent().height, 0.0f
    );

    vkCmdPushConstants(commandBuffers[imageIndex], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constant), &push_constant);
    
    ui->draw(commandBuffers[imageIndex]);

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

void UiApp::loadUi()
{
    ui = std::make_unique<UiRenderer>(device, 1000);
}

UiApp::~UiApp()
{
    vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
    std::free(clayMem);
}

void UiApp::buildUi() 
{
    
    Clay_SetLayoutDimensions((Clay_Dimensions) {static_cast<float>(window.getWidth()), static_cast<float>(window.getHeight())});

    Clay_BeginLayout();

    CLAY({ .id = CLAY_ID("background"), .layout = { .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)}, .padding = {.left = 30, .right = 30, .top = 30, .bottom = 30}, .childGap = 16, .layoutDirection = CLAY_TOP_TO_BOTTOM }, .backgroundColor = {200, 30, 200, 255} }) {
        CLAY({})
        for (int i = 0; i < 8; i++)
        {
            CLAY({.id = CLAY_IDI("Item", i), .layout = {.sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_PERCENT(0.15)}}, .backgroundColor = {(float)(10 + 20 * i), (float)(10 + 40 * i), (float)(100 + 30 * i),255} , .cornerRadius = { 30, 30, 30, 30 }}) {}
        }
    }

    Clay_RenderCommandArray renderCommands = Clay_EndLayout();

    // PrintRenderCommandArray(renderCommands);

    std::vector<RectangleItem> items;
    items.reserve(renderCommands.length);
    for (uint32_t i = 0; i < renderCommands.length; ++i) {
        const Clay_RenderCommand& rc = renderCommands.internalArray[i];
        if (rc.commandType != CLAY_RENDER_COMMAND_TYPE_RECTANGLE) continue;

        RectangleItem it{};
        const auto& bb = rc.boundingBox;
        const auto& rd = rc.renderData.rectangle;

        it.instance.position = { (float)bb.x, (float)bb.y };
        it.instance.size     = { (float)bb.width, (float)bb.height };

        auto N = [](float u8){ return u8 / 255.0f; };
        it.instance.color  = { N(rd.backgroundColor.r), N(rd.backgroundColor.g),
                           N(rd.backgroundColor.b), N(rd.backgroundColor.a) };

        it.instance.radius = { rd.cornerRadius.topLeft,  rd.cornerRadius.topRight,
                           rd.cornerRadius.bottomRight, rd.cornerRadius.bottomLeft };

        it.zIndex = rc.zIndex;
        it.seq = i;
        items.push_back(it);
    }

    // Painter’s algorithm: lowest z first (drawn first), tie breaks by command order
    std::stable_sort(items.begin(), items.end(),
        [](const RectangleItem& a, const RectangleItem& b){
            if (a.zIndex != b.zIndex) return a.zIndex < b.zIndex;
            return a.seq < b.seq;
        });

    rects.resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) rects[i] = items[i].instance;
}

void UiApp::run() {
    
    while (!window.shouldClose()) 
    {
        glfwPollEvents();
        drawFrame();
    }

    vkDeviceWaitIdle(device.device());
    
}