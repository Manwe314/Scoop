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
#include <printf.h>

#define CLAY_IMPLEMENTATION
#include "clay.h"
#include "clay_printer.h"

#include "UiApp.hpp"

static inline void ClayHandleErrors(Clay_ErrorData e) {
    fprintf(stderr, "[Clay] %.*s\n", (int)e.errorText.length, e.errorText.chars);
}

static Clay_Dimensions MeasureTextFromAtlas(Clay_StringSlice s, Clay_TextElementConfig* cfg, void* user)
{
    auto* tr = static_cast<TextRenderer*>(user);
    const int px = tr->getBucketPx((int)cfg->fontSize);      // match renderer bucket
    const auto* A = tr->getAtlasForPx(px);
    if (!A) return {0, 0};

    float ls = (float)cfg->letterSpacing;
    float lineH = (cfg->lineHeight > 0.0f) ? (float)cfg->lineHeight : (A->ascent - A->descent + A->lineGap);

    float w = 0.0f, maxW = 0.0f;
    // simple newline-aware width (Clay’s CLAY_TEXT doesn’t soft-wrap on width)
    for (int i = 0; i < s.length; ++i) {
        unsigned char c = (unsigned char)s.chars[i];
        if (c == '\n') { maxW = std::max(maxW, w); w = 0.0f; continue; }
        if (c < 32 || c >= 127) continue;
        const auto& g = A->glyphs[c];
        w += g.advance + ls;
    }
    maxW = std::max(maxW, w);

    // height = lines * lineHeight
    int lines = 1;
    for (int i = 0; i < s.length; ++i) if (s.chars[i] == '\n') ++lines;
    float h = lines * lineH;

    return { maxW, h };
}

// helper (accurate sRGB -> linear)
static inline float srgbToLinear(float c) {
    if (c <= 0.04045f) return c / 12.92f;
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

static inline glm::vec4 srgbToLinear(glm::vec4 c) {
    return { srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b), c.a };
}

inline Clay_String ClayFrom(const std::string& s) {
    Clay_String out;
    out.length = static_cast<int32_t>(s.size());
    out.chars  = const_cast<char*>(s.data());   // Clay never mutates; safe
    return out;
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
    createTextPipelineLayout();
    recreateSwapchain();
    createCommandBuffers();
    clayMemSize = Clay_MinMemorySize();
    clayMem = std::malloc(clayMemSize);
    if (!clayMem) throw std::runtime_error("Failed to alloc Clay memory");
    clayArena   = Clay_CreateArenaWithCapacityAndMemory(clayMemSize, clayMem);

    int w = WIDTH, h = HEIGHT;
    Clay_Initialize(clayArena, { (float)w, (float)h }, { ClayHandleErrors });
    
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

void UiApp::createTextPipelineLayout() {
    VkDescriptorSetLayout set0 = text->getDescriptorSetLayout();

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.offset = 0;
    range.size   = sizeof(SimplePushConstantData); // mat4 uProj

    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.setLayoutCount = 1;
    info.pSetLayouts    = &set0;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges    = &range;

    if (vkCreatePipelineLayout(device.device(), &info, nullptr, &textPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create text pipeline layout");
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

void UiApp::createTextPipeline() {
    assert(swapChain && textPipelineLayout);

    PipelineConfigInfo cfg{};
    Pipeline::defaultPipelineConfigIngo(cfg);
    cfg.renderPass     = swapChain->getRenderPass();
    cfg.pipelineLayout = textPipelineLayout;

    // Same tweaks as rects
    cfg.inputAssemblyInfo.topology       = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    cfg.depthStencilInfo.depthTestEnable = VK_FALSE;
    cfg.depthStencilInfo.depthWriteEnable= VK_FALSE;

    // Premultiplied blending
    cfg.colorBlendAttachment.blendEnable           = VK_TRUE;
    cfg.colorBlendAttachment.srcColorBlendFactor   = VK_BLEND_FACTOR_ONE;
    cfg.colorBlendAttachment.dstColorBlendFactor   = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cfg.colorBlendAttachment.colorBlendOp          = VK_BLEND_OP_ADD;
    cfg.colorBlendAttachment.srcAlphaBlendFactor   = VK_BLEND_FACTOR_ONE;
    cfg.colorBlendAttachment.dstAlphaBlendFactor   = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cfg.colorBlendAttachment.alphaBlendOp          = VK_BLEND_OP_ADD;

    // Vertex input: binding 0 (quad), binding 1 (glyphs)
    auto b0 = UiRenderer::QuadVertex::getBindDescriptions();
    auto a0 = UiRenderer::QuadVertex::getAttributeDescriptions();
    auto b1 = TextRenderer::GlyphInstance::getBindDescriptions();
    auto a1 = TextRenderer::GlyphInstance::getAttributeDescriptions();

    std::vector<VkVertexInputBindingDescription> bindings;
    bindings.push_back(b0[0]);
    bindings.push_back(b1[0]);

    std::vector<VkVertexInputAttributeDescription> attrs;
    attrs.insert(attrs.end(), a0.begin(), a0.end());
    attrs.insert(attrs.end(), a1.begin(), a1.end());

    // for (auto& a : attrs)
    //   printf("attr loc=%u bind=%u fmt=%u offset=%u\n", a.location, a.binding, a.format, a.offset);
    // for (auto& b : bindings)
    //   printf("bind=%u stride=%u rate=%u\n", b.binding, b.stride, b.inputRate);


    textPipeline = std::make_unique<Pipeline>(
        device, cfg, bindings, attrs,
        "build/shaders/text.vert.spv",
        "build/shaders/text.frag.spv"
    );
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
    createTextPipeline();
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
    ui->updateInstances(rects);
    ui->bind(commandBuffers[imageIndex]);

    SimplePushConstantData push_constant{};
    push_constant.uProj = glm::ortho(
        0.0f, (float)swapChain->getSwapChainExtent().width,
        0.0f, (float)swapChain->getSwapChainExtent().height
    );

    vkCmdPushConstants(commandBuffers[imageIndex], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constant), &push_constant);
    
    ui->draw(commandBuffers[imageIndex]);

    if (!textBatches.empty()) {
        textPipeline->bind(commandBuffers[imageIndex]);
        vkCmdPushConstants(commandBuffers[imageIndex], textPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constant), &push_constant);

        // ---- Build one big vector and remember ranges per px ----
        // collect keys (pixel buckets) in a stable order
        std::vector<int> keys;
        keys.reserve(textBatches.size());
        for (auto& kv : textBatches) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end()); 

        // compute total glyphs + concatenation
        size_t total = 0;
        for (int px : keys) total += textBatches[px].size();

        // (optional) make sure the instance buffer is big enough
        text->ensureInstanceCapacity(static_cast<uint32_t>(total));

        std::vector<TextRenderer::GlyphInstance> all;
        all.reserve(total);

        // px -> (firstInstance offset, count)
        std::unordered_map<int, std::pair<uint32_t,uint32_t>> ranges;
        ranges.reserve(keys.size());

        uint32_t cursor = 0;
        for (int px : keys) {
            auto& batch = textBatches[px];
            uint32_t cnt = static_cast<uint32_t>(batch.size());
            if (cnt == 0) { ranges[px] = {cursor, 0}; continue; }

            ranges[px] = {cursor, cnt};
            all.insert(all.end(), batch.begin(), batch.end());
            cursor += cnt;
        }

        // ---- ONE upload for all glyph instances ----
        text->updateInstances(all);

        // ---- Bind VBs + draw each bucket with firstInstance selecting the slice ----
        // We’ll reuse text->bind to bind VBs and the right atlas descriptor for each px.
        for (int px : keys) {
            auto [first, count] = ranges[px];
            if (count == 0) continue;

            text->bind(commandBuffers[imageIndex], textPipelineLayout, px); // binds quadVB+instVB and atlas for this px
            text->drawRange(commandBuffers[imageIndex], count, first);
        }
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

void UiApp::loadUi()
{
    text = std::make_unique<TextRenderer>(device, device.getCommandPool(), device.graphicsQueue(), "assets/fonts/FacultyGlyphic-Regular.ttf");
    text->initGeometry(8000);
    ui = std::make_unique<UiRenderer>(device, 1000);
}

UiApp::~UiApp()
{
    vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
    vkDestroyPipelineLayout(device.device(), textPipelineLayout, nullptr);
    std::free(clayMem);
}

void UiApp::buildUi() 
{
    
    Clay_SetLayoutDimensions((Clay_Dimensions) {static_cast<float>(window.getWidth()), static_cast<float>(window.getHeight())});
    Clay_SetMeasureTextFunction(MeasureTextFromAtlas, text.get());
    std::string name = device.getName();

    Clay_BeginLayout();

    CLAY({ .id = CLAY_ID("background"), 
            .layout = { 
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)}, 
                .padding = {.left = 30, .right = 30, .top = 30, .bottom = 30}, 
                .childGap = 16, 
                .layoutDirection = CLAY_TOP_TO_BOTTOM 
            },
            .backgroundColor = {71, 50, 128, 255}
        }) {
        CLAY({ .id = CLAY_ID("head sign"),
                .layout = { 
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT()}, 
                    .padding = CLAY_PADDING_ALL(16),
                    .childAlignment = { .x = CLAY_ALIGN_X_LEFT, .y = CLAY_ALIGN_Y_CENTER },
                    .layoutDirection = CLAY_TOP_TO_BOTTOM
                }, 
                .backgroundColor = {51, 30, 108, 170}, 
                .cornerRadius = CLAY_CORNER_RADIUS(20)
                }){
            CLAY_TEXT(CLAY_STRING("Welcome To Scoop"), 
                CLAY_TEXT_CONFIG({ 
                    .textColor = {255, 243, 232, 255}, 
                    .fontSize = 128, 
                    .letterSpacing = 1, 
                    .wrapMode = CLAY_TEXT_WRAP_NONE, 
                    .textAlignment = CLAY_TEXT_ALIGN_CENTER
                }));
        }
        CLAY({ .id = CLAY_ID("selected device"),
                .layout = {
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)},
                    .padding = CLAY_PADDING_ALL(16),
                    .childGap = 16,
                    .layoutDirection = CLAY_LEFT_TO_RIGHT
                },
                .backgroundColor = {51, 30, 108, 170}, 
                .cornerRadius = CLAY_CORNER_RADIUS(20)
        }){
            CLAY_TEXT(CLAY_STRING("Selected Device: "), 
            CLAY_TEXT_CONFIG({
                .textColor = {255, 243, 232, 255}, 
                .fontSize = 64, 
                .letterSpacing = 1, 
                .wrapMode = CLAY_TEXT_WRAP_NONE, 
                .textAlignment = CLAY_TEXT_ALIGN_CENTER
            }));
            CLAY({.layout = {.sizing = {CLAY_SIZING_GROW(0)}}});
            CLAY_TEXT(ClayFrom(name), 
            CLAY_TEXT_CONFIG({
                .textColor = {255, 243, 232, 255}, 
                .fontSize = 64, 
                .letterSpacing = 1, 
                .wrapMode = CLAY_TEXT_WRAP_NONE, 
                .textAlignment = CLAY_TEXT_ALIGN_CENTER
            }));
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

        auto C = [](uint8_t u8){ return u8 / 255.0f; };            // raw sRGB
        glm::vec4 srgb = { C(rd.backgroundColor.r), C(rd.backgroundColor.g),
                           C(rd.backgroundColor.b), C(rd.backgroundColor.a) };
        it.instance.color = srgbToLinear(srgb); 

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

    textBatches.clear();

    for (uint32_t i = 0; i < renderCommands.length; ++i) {
        const Clay_RenderCommand& rc = renderCommands.internalArray[i];
        if (rc.commandType != CLAY_RENDER_COMMAND_TYPE_TEXT) continue;

        const auto& bb = rc.boundingBox;
        const auto& td = rc.renderData.text;

        // Pick the nearest baked size (16,24,32,64,128)
        int requestedPx = (int)td.fontSize;         // Clay's size is in pixels
        int bucketPx    = text->getBucketPx(requestedPx);

        // Convert Clay string -> std::string_view
        // Clay exposes length + chars; the exact field is 'td.text' with .chars/.length
        std::string_view txt{
            td.stringContents.chars,
            static_cast<std::size_t>(td.stringContents.length)
        };

        auto C = [](uint8_t u8){ return u8 / 255.0f; };
        glm::vec4 srgb = { C(td.textColor.r), C(td.textColor.g), C(td.textColor.b), C(td.textColor.a) };
        if (srgb.a == 0.0f) srgb.a = 1.0f;
        glm::vec4 linear = srgbToLinear(srgb);
        
        // Letter spacing & line height (Clay gives pixel values; 0 -> use font default)
        float ls  = (float)td.letterSpacing;
        float lh  = (float)td.lineHeight;  // if 0, TextRenderer uses ascent/descent/lineGap
        
        // Starting position is the top-left of the command's bbox
        glm::vec2 start = { (float)bb.x, (float)bb.y };
        
        // Layout glyphs for this text run using the atlas bucket
        auto& batch = textBatches[bucketPx];
        auto glyphs = text->layoutASCII(bucketPx, txt, start, linear, ls, lh);

        // Append to this bucket's vector
        batch.insert(batch.end(), glyphs.begin(), glyphs.end());
    }
}

void UiApp::run() {
    
    while (!window.shouldClose()) 
    {
        glfwPollEvents();
        drawFrame();
    }

    vkDeviceWaitIdle(device.device());
    
}