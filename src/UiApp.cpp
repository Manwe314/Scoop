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
    out.chars  = const_cast<char*>(s.c_str());
    return out;
}

inline Clay_String ClayFromStable(std::vector<std::string>& store, const std::string& s) {
    store.emplace_back(s);
    Clay_String out{};
    out.isStaticallyAllocated = false;
    out.length = static_cast<int32_t>(store.back().size());
    out.chars  = store.back().c_str();
    return out;
}

static std::vector<float> penStops(const TextRenderer& tr, int px, std::string_view txt, float startX, float letterSpacing)
{
    const auto* A = tr.getAtlasForPx(px);
    std::vector<float> stops;
    stops.reserve(txt.size()+1);
    float x = startX;
    stops.push_back(x);
    for (unsigned char c : txt) {
        if (c < 32 || c >= 127) { stops.push_back(x); continue; }
        x += A->glyphs[c].advance + letterSpacing;
        stops.push_back(x);
    }
    return stops;
}

static inline VkRect2D fullScissor(VkExtent2D fb) {
    return VkRect2D{ {0,0}, {fb.width, fb.height} };
}

static inline VkRect2D bboxToScissor(const Clay_BoundingBox& bb, VkExtent2D fb, int padY = 1) {
    int32_t x = (int32_t)bb.x;
    int32_t y = (int32_t)bb.y - padY;
    int32_t w = (int32_t)bb.width;
    int32_t h = (int32_t)bb.height + padY*2;
    // clamp
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    uint32_t W = (uint32_t)std::max(0, std::min((int)fb.width  - x, w));
    uint32_t H = (uint32_t)std::max(0, std::min((int)fb.height - y, h));
    return VkRect2D{ {x, y}, {W, H} };
}

static inline VkRect2D intersectScissor(VkRect2D a, VkRect2D b) {
    int ax0 = a.offset.x, ay0 = a.offset.y;
    int ax1 = ax0 + (int)a.extent.width;
    int ay1 = ay0 + (int)a.extent.height;

    int bx0 = b.offset.x, by0 = b.offset.y;
    int bx1 = bx0 + (int)b.extent.width;
    int by1 = by0 + (int)b.extent.height;

    int x0 = std::max(ax0, bx0);
    int y0 = std::max(ay0, by0);
    int x1 = std::min(ax1, bx1);
    int y1 = std::min(ay1, by1);

    if (x1 <= x0 || y1 <= y0) return VkRect2D{ {0,0}, {0,0} };
    return VkRect2D{ {x0, y0}, { (uint32_t)(x1-x0), (uint32_t)(y1-y0) } };
}



struct SimplePushConstantData {
    glm::mat4 uProj;
};

struct RectangleItem {
    UiRenderer::RectangleInstance instance;
    int zIndex;
    uint32_t seq;
};

UiApp::UiApp(std::string def, VulkanContext& context) : window(WIDTH, HEIGHT, "UI"), device(window, context.getInstance()), clayFrameStrings()
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
    
    s_active = this;
    glfwSetCharCallback(window.handle(), &UiApp::CharCallback);
    glfwSetKeyCallback (window.handle(), &UiApp::KeyCallback);
    cursorArrow = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    cursorHand  = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
    cursorIBeam = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
    wanted = cursorArrow;
    std::cout << def << std::endl;
    input.text = def; 
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

UiApp* UiApp::s_active = nullptr;

void UiApp::CharCallback(GLFWwindow*, unsigned int cp) {
    if (s_active) s_active->onChar(cp);
}

void UiApp::KeyCallback(GLFWwindow*, int key, int, int action, int mods) {
    if (s_active) s_active->onKey(key, action, mods);
}

void UiApp::onChar(uint32_t cp) {
    if (!input.focused) return;

    // Basic printable ASCII range (expand later to UTF-8 if you like)
    if (cp >= 32 && cp < 127) {
        input.text.push_back(static_cast<char>(cp));
        input.caret = input.text.size();
        input.blinkStart = glfwGetTime();
        std::cout << "[Input] text=\"" << input.text << "\"\n";
    }
}

void UiApp::onKey(int key, int action, int /*mods*/) {
    if (!input.focused) return;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    switch (key) {
        case GLFW_KEY_BACKSPACE:
            if (!input.text.empty()) {
                input.text.pop_back();
                input.caret = input.text.size();
                input.blinkStart = glfwGetTime();
                std::cout << "[Input] text=\"" << input.text << "\"\n";
            }
            break;
        case GLFW_KEY_ENTER:
        case GLFW_KEY_KP_ENTER:
            std::cout << "[Input] submit=\"" << input.text << "\"\n";
            break;
        case GLFW_KEY_ESCAPE:
            input.focused = false;
            std::cout << "[Input] unfocused (Esc)\n";
            break;
        default:
            break;
    }
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


    if (!tempRuns.empty()) {
        textPipeline->bind(commandBuffers[imageIndex]);
        vkCmdPushConstants(commandBuffers[imageIndex], textPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constant), &push_constant);

        size_t total = 0;
        for (auto& r : tempRuns) total += r.glyphs.size();

        text->ensureInstanceCapacity((uint32_t)total);

        std::vector<TextRenderer::GlyphInstance> all;
        all.reserve(total);

        struct TextRun { int px; VkRect2D sc; uint32_t first, count; int z; uint32_t seq; };
        std::vector<TextRun> runs; runs.reserve(tempRuns.size());

        uint32_t cursor = 0;
        for (auto& r : tempRuns) {
            TextRun out{ r.px, r.scissor, cursor, (uint32_t)r.glyphs.size(), r.z, r.seq };
            all.insert(all.end(), r.glyphs.begin(), r.glyphs.end());
            cursor += out.count;
            runs.push_back(out);
        }

        // painter order
        std::stable_sort(runs.begin(), runs.end(),
            [](const TextRun& a, const TextRun& b){
                if (a.z != b.z) return a.z < b.z;
                return a.seq < b.seq;
            });

        text->updateInstances(all);

        for (const auto& run : runs) {
            if (run.count == 0) continue;
            vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &run.sc);
            text->bind(commandBuffers[imageIndex], textPipelineLayout, run.px);
            text->drawRange(commandBuffers[imageIndex], run.count, run.first);
        }

        // restore full scissor for subsequent draws (caret, etc.)
        VkRect2D full{{0,0}, swapChain->getSwapChainExtent()};
        vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &full);
        if (input.focused) {
            // Blink: 0.5s on, 0.5s off
            const double now = glfwGetTime();
            const bool show = fmod(now - input.blinkStart, 1.0) < 0.5;
            if (show && focusedInputRect.width > 0) {
                const float pad = 10.0f; // must match input box padding
                const float startX = (float)focusedInputRect.x + pad;
                const float startY = (float)focusedInputRect.y + pad;
            
                // caret index = end (for now)
                const size_t caretIdx = std::min(input.caret, input.text.size());
                auto stops = penStops(*text, input.fontPx, input.text, startX, input.letterSpacing);
                const float caretX = (caretIdx < stops.size()) ? stops[caretIdx] : stops.back();
            
                // Clip to the input rect
                VkRect2D inputScissor{
                    { (int32_t)focusedInputRect.x, (int32_t)focusedInputRect.y },
                    { (uint32_t)focusedInputRect.width, (uint32_t)focusedInputRect.height }
                };
                vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &inputScissor);
            
                // Build a single rectangle instance as the caret
                UiRenderer::RectangleInstance caret{};
                caret.position = { caretX, (float)focusedInputRect.y + 8.0f };
                caret.size     = { 2.0f, (float)focusedInputRect.height - 20.0f }; // thin bar
                caret.color    = srgbToLinear({0.1f,0.1f,0.1f,1});
                caret.radius   = {0,0,0,0};
            
                // Bind rect pipeline again (we just used text pipeline)
                pipeline->bind(commandBuffers[imageIndex]);
            
                // Push constants for this pipeline layout again
                SimplePushConstantData pc{};
                pc.uProj = glm::ortho(0.0f, (float)swapChain->getSwapChainExtent().width,
                                      0.0f, (float)swapChain->getSwapChainExtent().height);
                vkCmdPushConstants(commandBuffers[imageIndex], pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                
                // Upload only the caret and draw it
                UiRenderer::RectangleInstance one = caret;
                uiOverlay->updateInstances(std::vector<UiRenderer::RectangleInstance>{ one });
                uiOverlay->bind(commandBuffers[imageIndex]);
                uiOverlay->draw(commandBuffers[imageIndex]);
                
                // Restore full scissor
                VkRect2D full{{0,0}, swapChain->getSwapChainExtent()};
                vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &full);
            }
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
    uiOverlay = std::make_unique<UiRenderer>(device, 12);
}

void UiApp::reconstruct()
{
    swapChain.reset();
    window.recreate();
    device.createSurface();
    recreateSwapchain();
    glfwSetCharCallback(window.handle(), &UiApp::CharCallback);
    glfwSetKeyCallback (window.handle(), &UiApp::KeyCallback);
}

UiApp::~UiApp()
{
    vkDestroyPipelineLayout(device.device(), pipelineLayout, nullptr);
    vkDestroyPipelineLayout(device.device(), textPipelineLayout, nullptr);
    std::free(clayMem);
    glfwDestroyCursor(cursorArrow);
    glfwDestroyCursor(cursorHand);
    glfwDestroyCursor(cursorIBeam);
}

void UiApp::HandleButtonInteraction(Clay_ElementId elementId, Clay_PointerData pointerData) 
{

    if (elementId.id == Clay_GetElementId(CLAY_STRING("input form")).id)
        wanted = cursorIBeam;
    else if (elementId.id != Clay_GetElementId(CLAY_STRING("background")).id)
        wanted = cursorHand;

    
    if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("input form")).id)) {
        input.focused = true;
        input.caret = input.text.size();
        focusedInputId = elementId;
        input.blinkStart = glfwGetTime();
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("background")).id)) {
        input.focused = false;
        focusedInputId = Clay_ElementId{0};
        // try
        // {
        //     std::string filePath("./assets/models/teapot.obj");
        //     Object obj(filePath);
        //     SBVH sbvh = obj.buildSplitBoundingVolumeHierarchy();
        //     std::cout << "the size of nodes: " << (int)sbvh.nodes.size() << std::endl;
        // }
        // catch(const std::exception& e)
        // {
        //     std::cerr << e.what() << '\n';
        // }
        
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME) {
        std::cout << "BUTTON CLICKED!" << std::endl;
        state.shouldClose = false;
        state.device = device.getPhysicalDevice();
        exitRun = true;
        input.focused = false;
        focusedInputId = Clay_ElementId{0};
    }
}

void UiApp::buildUi() 
{
    
    Clay_SetLayoutDimensions((Clay_Dimensions) {static_cast<float>(window.getWidth()), static_cast<float>(window.getHeight())});
    Clay_SetMeasureTextFunction(MeasureTextFromAtlas, text.get());
    std::string name = device.getName();
    clayFrameStrings.clear();
    wanted = cursorArrow;
    
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
        Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
        CLAY({ .id = CLAY_ID("head sign"),
                .layout = { 
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT()}, 
                    .padding = CLAY_PADDING_ALL(16),
                    .childAlignment = { .x = CLAY_ALIGN_X_LEFT, .y = CLAY_ALIGN_Y_CENTER },
                    .layoutDirection = CLAY_TOP_TO_BOTTOM
                }, 
                .backgroundColor = {51, 30, 108, 170}, 
                .cornerRadius = CLAY_CORNER_RADIUS(20),
                .clip = {.horizontal = true, .vertical = true}
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
                .cornerRadius = CLAY_CORNER_RADIUS(20),
                .clip = {.horizontal = true, .vertical = true}
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
            CLAY_TEXT(ClayFromStable(clayFrameStrings ,name), 
            CLAY_TEXT_CONFIG({
                .textColor = {255, 243, 232, 255}, 
                .fontSize = 64, 
                .letterSpacing = 1, 
                .wrapMode = CLAY_TEXT_WRAP_NONE, 
                .textAlignment = CLAY_TEXT_ALIGN_CENTER
            }));
        }
        CLAY({ .id = CLAY_ID("button"),
                .layout = {
                    .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                    .padding = CLAY_PADDING_ALL(10),
                },
                .backgroundColor = {65, 9, 114, 200},
                .cornerRadius = CLAY_CORNER_RADIUS(35),
                .clip = {.horizontal = true, .vertical = true}
        }){
            Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
            CLAY_TEXT(CLAY_STRING("Button"), 
            CLAY_TEXT_CONFIG({
                .textColor = {255, 243, 232, 255}, 
                .fontSize = 32, 
                .letterSpacing = 1, 
                .wrapMode = CLAY_TEXT_WRAP_NONE, 
                .textAlignment = CLAY_TEXT_ALIGN_CENTER
            }));
        }
        CLAY({ .id = CLAY_ID("form"),
                .layout = {
                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)},
                    .padding = CLAY_PADDING_ALL(16),
                    .childGap = 16,
                    .layoutDirection = CLAY_TOP_TO_BOTTOM
                },
                .backgroundColor = {51, 30, 108, 170}, 
                .cornerRadius = CLAY_CORNER_RADIUS(20),
                .clip = {.horizontal = true, .vertical = true}
        }){
            CLAY_TEXT(CLAY_STRING("Relative Path to Model"), 
            CLAY_TEXT_CONFIG({
                .textColor = {255, 243, 232, 255}, 
                .fontSize = 32, 
                .letterSpacing = 1, 
                .wrapMode = CLAY_TEXT_WRAP_NONE, 
                .textAlignment = CLAY_TEXT_ALIGN_CENTER
            }));
            CLAY({ .id = CLAY_ID("text form"),
                    .layout = {
                        .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)},
                        .childGap = 16,
                        .layoutDirection = CLAY_LEFT_TO_RIGHT
                    }
            }){
                CLAY({ .id = CLAY_ID("input form"),
                        .layout = {
                            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                            .padding = CLAY_PADDING_ALL(10)
                        },
                        .backgroundColor = {179, 170, 162, 170}, 
                        .cornerRadius = CLAY_CORNER_RADIUS(20),
                        .clip = {.horizontal = true, .vertical = true}
                }){
                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                    CLAY_TEXT(ClayFromStable(clayFrameStrings, input.text),
                    CLAY_TEXT_CONFIG({
                        .textColor = {255, 243, 232, 255},
                        .fontSize  = 32,
                        .letterSpacing = 1,
                        .wrapMode  = CLAY_TEXT_WRAP_NONE,
                        .textAlignment = CLAY_TEXT_ALIGN_LEFT
                    }));
                }
                CLAY({ .id = CLAY_ID("model loader"),
                        .layout = {
                            .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                            .padding = CLAY_PADDING_ALL(10),
                        },
                        .backgroundColor = {65, 9, 114, 200},
                        .cornerRadius = CLAY_CORNER_RADIUS(35),
                        .clip = {.horizontal = true, .vertical = true}
                }){
                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                    CLAY_TEXT(CLAY_STRING("Load Model"), 
                    CLAY_TEXT_CONFIG({
                        .textColor = {255, 243, 232, 255}, 
                        .fontSize = 32, 
                        .letterSpacing = 1, 
                        .wrapMode = CLAY_TEXT_WRAP_NONE, 
                        .textAlignment = CLAY_TEXT_ALIGN_CENTER
                    }));
                }
            }
        }
    }

    Clay_RenderCommandArray renderCommands = Clay_EndLayout();

    {
        // Get cursor in window coords (origin top-left, y down)
        double mx, my;
        window.getCursorPos(mx, my);

        // Convert to framebuffer pixel coords if you’re using framebuffer sizes for layout
        int winW, winH, fbW, fbH;
        window.getSizes(winW, winH, fbW, fbH);
        float sx = (winW > 0) ? (float)fbW / (float)winW : 1.0f;
        float sy = (winH > 0) ? (float)fbH / (float)winH : 1.0f;

        Clay_Vector2 p = { (float)mx * sx, (float)my * sy };

        bool leftDown = window.isMouseDown(GLFW_MOUSE_BUTTON_LEFT);
        Clay_SetPointerState(p, leftDown);
    }

    // PrintRenderCommandArray(renderCommands);
    glfwSetCursor(window.handle(), wanted);

    textRuns.clear();
    textRuns.reserve(renderCommands.length);

   
    std::vector<RunTemp> runTemps;
    runTemps.reserve(renderCommands.length);

    std::vector<RectangleItem> items;
    items.reserve(renderCommands.length);
    for (uint32_t i = 0; i < renderCommands.length; ++i) {
        const Clay_RenderCommand& rc = renderCommands.internalArray[i];
        if (rc.commandType != CLAY_RENDER_COMMAND_TYPE_RECTANGLE) continue;

        if (rc.id == focusedInputId.id) {
            focusedInputRect = rc.boundingBox;
        }

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

    VkExtent2D fb = swapChain->getSwapChainExtent();
    std::vector<VkRect2D> scissorStack;
    scissorStack.push_back(VkRect2D{{0,0},{fb.width, fb.height}}); // full

    for (uint32_t i = 0; i < renderCommands.length; ++i) {
        const Clay_RenderCommand& rc = renderCommands.internalArray[i];

        if (rc.commandType == CLAY_RENDER_COMMAND_TYPE_SCISSOR_START) {
            // Build a scissor from this bbox, pad Y to avoid cutting descenders
            VkRect2D clipHere = bboxToScissor(rc.boundingBox, fb, /*padY*/1);

            // If Clay exposes rc.clip.{horizontal,vertical}, widen per axis when disabled
            // (so a vertical-only clip doesn't restrict X, etc.)
            const Clay_ClipRenderData& clipCfg = rc.renderData.clip;
            if (!clipCfg.horizontal) { clipHere.offset.x = 0; clipHere.extent.width  = fb.width;  }
            if (!clipCfg.vertical)   { clipHere.offset.y = 0; clipHere.extent.height = fb.height; }

            VkRect2D combined = intersectScissor(scissorStack.back(), clipHere);
            scissorStack.push_back(combined);
            continue;
        }

        if (rc.commandType == CLAY_RENDER_COMMAND_TYPE_SCISSOR_END) {
            if (scissorStack.size() > 1) scissorStack.pop_back();
            continue;
        }

        if (rc.commandType != CLAY_RENDER_COMMAND_TYPE_TEXT) continue;

        const auto& bb = rc.boundingBox;
        const auto& td = rc.renderData.text;

        const int requestedPx = (int)td.fontSize;
        const int bucketPx    = text->getBucketPx(requestedPx);

        std::string_view txt{ td.stringContents.chars, (size_t)td.stringContents.length };

        auto Csrgb = [](uint8_t u8){ return u8 / 255.0f; };
        glm::vec4 srgb = { Csrgb(td.textColor.r), Csrgb(td.textColor.g),
                           Csrgb(td.textColor.b), Csrgb(td.textColor.a) };
        if (srgb.a == 0.0f) srgb.a = 1.0f;
        glm::vec4 linear = srgbToLinear(srgb);

        float ls  = (float)td.letterSpacing;
        float lh  = (float)td.lineHeight;

        glm::vec2 start = { (float)bb.x, (float)bb.y };

        RunTemp r{};
        r.px      = bucketPx;
        r.scissor = scissorStack.back(); // <-- ACTIVE CLIP FROM CLAY
        r.glyphs  = text->layoutASCII(bucketPx, txt, start, linear, ls, lh);
        r.z       = rc.zIndex;
        r.seq     = i;

        runTemps.push_back(std::move(r));
    }

    std::stable_sort(runTemps.begin(), runTemps.end(),
        [](const RunTemp& a, const RunTemp& b){
            if (a.z != b.z) return a.z < b.z;
            return a.seq < b.seq;
        });
    this->tempRuns = std::move(runTemps);
}

AppState UiApp::run()
{
    state.shouldClose = true;
    state.device = VK_NULL_HANDLE;
    if (window.handle() == nullptr)
        std::cout << "WHOOOPS its Null" << std::endl;

    if (exitRun == true)
    {
        reconstruct();
        exitRun = false;
    }
    while (!window.shouldClose()) 
    {
        glfwPollEvents();
        drawFrame();
        if (exitRun == true)
            break;
    }

    vkDeviceWaitIdle(device.device());
    glfwHideWindow(window.handle());
    return state;
}