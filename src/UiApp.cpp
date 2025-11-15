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
    //fprintf(stderr, "[Clay] %.*s\n", (int)e.errorText.length, e.errorText.chars);
}

static inline bool matchesBase(Clay_ElementId id, Clay_ElementId base) {
    return id.baseId == base.baseId;
}

static Clay_Dimensions MeasureTextFromAtlas(Clay_StringSlice s, Clay_TextElementConfig* cfg, void* user)
{
    auto* tr = static_cast<TextRenderer*>(user);
    const int px = tr->getBucketPx((int)cfg->fontSize);
    const auto* A = tr->getAtlasForPx(px);
    if (!A) return {0, 0};

    float ls = (float)cfg->letterSpacing;
    float lineH = (cfg->lineHeight > 0.0f) ? (float)cfg->lineHeight : (A->ascent - A->descent + A->lineGap);

    float w = 0.0f, maxW = 0.0f;
    for (int i = 0; i < s.length; ++i) {
        unsigned char c = (unsigned char)s.chars[i];
        if (c == '\n') { maxW = std::max(maxW, w); w = 0.0f; continue; }
        if (c < 32 || c >= 127) continue;
        const auto& g = A->glyphs[c];
        w += g.advance + ls;
    }
    maxW = std::max(maxW, w);

    int lines = 1;
    for (int i = 0; i < s.length; ++i) if (s.chars[i] == '\n') ++lines;
    float h = lines * lineH;

    return { maxW, h };
}

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

static inline int findMeshIndexByName(const Scene& scene, const std::string& name)
{
    for (size_t i = 0; i < scene.meshes.size(); ++i)
        if (scene.meshes[i].name == name) return (int)i;
    return -1;
}

std::optional<size_t> indexFromTabSid(const Sid sid, const std::vector<ModelTab>& models)
{
    for (size_t i = 0; i < models.size(); ++i)
        if (models[i].stableID == sid) return i;
    return std::nullopt;
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

static inline bool isObj(std::string& path)
{
    return path.size() >= 5 && path.compare(path.size() - 4, 4, ".obj") == 0;
}

static std::string pickObjRelativePath()
{
    if (NFD_Init() != NFD_OKAY)
        throw std::runtime_error("File Dialog Init Failed.");

    nfdfilteritem_t filters[] = {{ "OBJ Models", "obj" }};

    nfdchar_t* outPath = nullptr;
    const nfdresult_t res = NFD_OpenDialog(&outPath, filters, 1, nullptr); 

    std::string result;

    if (res == NFD_OKAY && outPath)
    {
        std::filesystem::path abs = outPath;
        free(outPath);

        std::error_code ec;
        std::filesystem::path base = std::filesystem::current_path();
        std::filesystem::path rel = std::filesystem::relative(abs, base, ec);

        result = (!ec && !rel.empty()) ? rel.string() : abs.string();
    }
    else if (res == NFD_ERROR)
        throw std::runtime_error("Unexpected Error wile using File dialog");

    NFD_Quit();
    return result;
}

static inline bool PressedThis(const Clay_PointerData& p)
{
    return p.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME;
}

bool UiApp::isModelButton(Clay_ElementId elementId)
{
    
    if (matchesBase(elementId, Clay_GetElementId(CLAY_STRING("model.build"))) ||
        matchesBase(elementId, Clay_GetElementId(CLAY_STRING("model.instance.add"))) ||
        matchesBase(elementId, Clay_GetElementId(CLAY_STRING("model.instance.del")))
    )
        return true;
    return false;
}

bool UiApp::isInput(Clay_ElementId elementId)
{
    for (auto& fieldId : inputFields)
        if (elementId.id == fieldId.id) return true;

    static const Clay_ElementId modelInputBases[] = {
        Clay_GetElementId(CLAY_STRING("model.translate.input.x")),
        Clay_GetElementId(CLAY_STRING("model.translate.input.y")),
        Clay_GetElementId(CLAY_STRING("model.translate.input.z")),
        Clay_GetElementId(CLAY_STRING("model.rotation.input.x")),
        Clay_GetElementId(CLAY_STRING("model.rotation.input.y")),
        Clay_GetElementId(CLAY_STRING("model.rotation.input.z")),
        Clay_GetElementId(CLAY_STRING("model.scale.input.x")),
        Clay_GetElementId(CLAY_STRING("model.scale.input.y")),
        Clay_GetElementId(CLAY_STRING("model.scale.input.z")),
        Clay_GetElementId(CLAY_STRING("model.rotspeed.input.x")),
        Clay_GetElementId(CLAY_STRING("model.rotspeed.input.y")),
        Clay_GetElementId(CLAY_STRING("model.rotspeed.input.z")),
        Clay_GetElementId(CLAY_STRING("model.speedscalar.input")),
    };

    for (const auto& base : modelInputBases)
        if (matchesBase(elementId, base)) return true;

    return false;
}

struct SimplePushConstantData {
    Mat4 uProj;
};

struct RectangleItem {
    UiRenderer::RectangleInstance instance;
    int zIndex;
    uint32_t seq;
};

UiApp::UiApp(std::string def, VulkanContext& context) : window(WIDTH, HEIGHT, "UI"), device(window, context.getInstance()), clayFrameStrings(), optionalDevices(), optionalDeviceNames()
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
    auto elementId = Clay_GetElementId(CLAY_STRING("input form"));
    inputs.get(elementId).text = def;
    clayFrameStrings.reserve(80);
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.position.input.x"))).text = "0";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.position.input.y"))).text = "0";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.position.input.z"))).text = "5";

    inputs.get(Clay_GetElementId(CLAY_STRING("cam.target.input.x"))).text = "0";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.target.input.y"))).text = "0";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.target.input.z"))).text = "0";

    inputs.get(Clay_GetElementId(CLAY_STRING("cam.up.input.x"))).text = "0";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.up.input.y"))).text = "1";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.up.input.z"))).text = "0";

    inputs.get(Clay_GetElementId(CLAY_STRING("cam.vfov.input"))).text     = "60";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.aspect.input"))).text   = "1.7778";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.near.input"))).text     = "0.1";
    inputs.get(Clay_GetElementId(CLAY_STRING("cam.far.input"))).text      = "1000";
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

void UiApp::CharCallback(GLFWwindow*, unsigned int cp)
{
    if (s_active)
        s_active->onChar(cp);
}

void UiApp::KeyCallback(GLFWwindow*, int key, int, int action, int mods)
{
    if (s_active)
        s_active->onKey(key, action, mods);
}

void UiApp::onChar(uint32_t cp)
{
    auto* s = inputs.focused();
    if (!s) return;

    if (cp >= 32 && cp < 127) {
        s->text.push_back(static_cast<char>(cp));
        s->caret = s->text.size();
        s->blinkStart = glfwGetTime();
    }
}

void UiApp::onKey(int key, int action, int )
{
    auto* s = inputs.focused();
    if (!s || (action != GLFW_PRESS && action != GLFW_REPEAT))
        return;

    switch (key) {
        case GLFW_KEY_BACKSPACE:
            if (!s->text.empty()) {
                s->text.pop_back();
                s->caret = s->text.size();
                s->blinkStart = glfwGetTime();
            }
            break;
        case GLFW_KEY_ENTER:
        case GLFW_KEY_KP_ENTER:
            break;
        case GLFW_KEY_ESCAPE:
            s->focused = false;
            break;
        default:
            break;
    }
}

// void UiApp::resetUiApp()
// {
//     for (auto& mesh : state.scene.meshes)
//     {

//     }

// }

void UiApp::createTextPipelineLayout()
{
    VkDescriptorSetLayout set0 = text->getDescriptorSetLayout();

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.offset = 0;
    range.size   = sizeof(SimplePushConstantData); 

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
    push_constant.uProj = ortho(
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

        VkRect2D full{{0,0}, swapChain->getSwapChainExtent()};
        vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &full);
        auto* s = inputs.focused();
        if (s != nullptr) {
            const double now = glfwGetTime();
            const bool show = fmod(now - s->blinkStart, 1.0) < 0.5;
            if (show && focusedInputRect.width > 0) {
                const float pad = 10.0f;
                const float startX = (float)focusedInputRect.x + pad;
                const float startY = (float)focusedInputRect.y + pad;
            
                const size_t caretIdx = std::min(s->caret, s->text.size());
                auto stops = penStops(*text, s->fontPx, s->text, startX, s->letterSpacing);
                const float caretX = (caretIdx < stops.size()) ? stops[caretIdx] : stops.back();
            
                VkRect2D inputScissor{
                    { (int32_t)focusedInputRect.x, (int32_t)focusedInputRect.y },
                    { (uint32_t)focusedInputRect.width, (uint32_t)focusedInputRect.height }
                };
                vkCmdSetScissor(commandBuffers[imageIndex], 0, 1, &inputScissor);
            
                UiRenderer::RectangleInstance caret{};
                caret.position = { caretX, (float)focusedInputRect.y + 8.0f };
                caret.size     = { 2.0f, (float)focusedInputRect.height - 20.0f }; // thin bar
                caret.color    = srgbToLinear({0.1f,0.1f,0.1f,1});
                caret.radius   = {0,0,0,0};
            
                pipeline->bind(commandBuffers[imageIndex]);
            
                SimplePushConstantData pc{};
                pc.uProj = ortho(0.0f, (float)swapChain->getSwapChainExtent().width,
                                 0.0f, (float)swapChain->getSwapChainExtent().height);
                vkCmdPushConstants(commandBuffers[imageIndex], pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                
                UiRenderer::RectangleInstance one = caret;
                uiOverlay->updateInstances(std::vector<UiRenderer::RectangleInstance>{ one });
                uiOverlay->bind(commandBuffers[imageIndex]);
                uiOverlay->draw(commandBuffers[imageIndex]);
                
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

void UiApp::finalizeSceneForLaunch()
{
    Scene& scene = state.scene;

    for (size_t i = 0; i < models.size(); ++i) {
        const std::string& name = models[i].name;

        uint32_t meshID = 0;
        for (uint32_t j = 0; j < scene.meshes.size(); ++j)
            if (scene.meshes[j].name == name)
            { 
                meshID = j;
                break;
            }

        SceneObject& obj = scene.objects[i];
        obj.meshID = meshID;

        obj.boundingBox = scene.meshes[meshID].bottomLevelAccelerationStructure.outerBoundingBox;
    }
}

void UiApp::UpdateInput(bool sameIndex)
{
    auto readFloat = [&](const char* key) -> std::optional<float> {
        Clay_String str{false, (int32_t)std::strlen(key), const_cast<char*>(key)};
        Clay_ElementId id = Clay_GetElementId(str);
        const std::string& s = inputs.get(id).text;
        return to_float(s);
    };

    auto readFloatById = [&](Clay_ElementId id) -> std::optional<float> {
        const std::string& s = inputs.get(id).text;
        return to_float(s);
    };

    if (!sameIndex && uiState.targetEditor != uiState.previousTargetEditor)
    {
        
        if (uiState.previousTargetEditor == -1)
        {

            Camera cam = state.scene.camera;

            if (auto v = readFloat("cam.position.input.x")) cam.position.x = *v;
            if (auto v = readFloat("cam.position.input.y")) cam.position.y = *v;
            if (auto v = readFloat("cam.position.input.z")) cam.position.z = *v;

            if (auto v = readFloat("cam.target.input.x"))   cam.target.x   = *v;
            if (auto v = readFloat("cam.target.input.y"))   cam.target.y   = *v;
            if (auto v = readFloat("cam.target.input.z"))   cam.target.z   = *v;

            if (auto v = readFloat("cam.up.input.x"))       cam.up.x       = *v;
            if (auto v = readFloat("cam.up.input.y"))       cam.up.y       = *v;
            if (auto v = readFloat("cam.up.input.z"))       cam.up.z       = *v;

            if (auto v = readFloat("cam.vfov.input"))       cam.vfovDeg    = *v;
            if (auto v = readFloat("cam.aspect.input"))     cam.aspect     = *v;
            if (auto v = readFloat("cam.near.input"))       cam.nearPlane  = *v;
            if (auto v = readFloat("cam.far.input"))        cam.farPlane   = *v;

            state.scene.camera = cam;
        }
        else if (uiState.previousTargetEditor >= 0)
        {
            const int prevIdx = uiState.previousTargetEditor;
        
            auto optSid = sidForIndex(models, prevIdx);
            if (!optSid) { uiState.previousTargetEditor = uiState.targetEditor; return; }
            Sid sid = *optSid;
        
            SceneObject objCopy{};
            if ((size_t)prevIdx < state.scene.objects.size())
                objCopy = state.scene.objects[prevIdx];
        
            Transform         transform = objCopy.transform;
            RotationAnimation anim      = objCopy.animation;
        
            auto rf = [&](const char* key) -> std::optional<float> {
                Clay_String str{false, (int32_t)std::strlen(key), const_cast<char*>(key)};
                Clay_ElementId id = Clay_GetElementIdWithIndex(str, sid);
                return readFloatById(id);
            };
        
            if (auto v = rf("model.translate.input.x")) transform.translate.x = *v;
            if (auto v = rf("model.translate.input.y")) transform.translate.y = *v;
            if (auto v = rf("model.translate.input.z")) transform.translate.z = *v;
        
            if (auto v = rf("model.rotation.input.x"))  transform.rotation.x  = *v;
            if (auto v = rf("model.rotation.input.y"))  transform.rotation.y  = *v;
            if (auto v = rf("model.rotation.input.z"))  transform.rotation.z  = *v;
        
            if (auto v = rf("model.scale.input.x"))     transform.scale.x     = *v;
            if (auto v = rf("model.scale.input.y"))     transform.scale.y     = *v;
            if (auto v = rf("model.scale.input.z"))     transform.scale.z     = *v;
        
            if (auto v = rf("model.rotspeed.input.x"))  anim.rotationSpeedPerAxis.x = *v;
            if (auto v = rf("model.rotspeed.input.y"))  anim.rotationSpeedPerAxis.y = *v;
            if (auto v = rf("model.rotspeed.input.z"))  anim.rotationSpeedPerAxis.z = *v;
        
            if (auto v = rf("model.speedscalar.input")) anim.SpeedScalar = *v;
        
            if ((size_t)prevIdx >= state.scene.objects.size())
                state.scene.objects.resize((size_t)prevIdx + 1);
        
            state.scene.objects[prevIdx].transform = transform;
            state.scene.objects[prevIdx].animation = anim;
        
            uiState.previousTargetEditor = uiState.targetEditor;
        }
        uiState.previousTargetEditor = uiState.targetEditor;
    }
    else if (sameIndex)
    {
        if (uiState.targetEditor == -1)
        {

            Camera cam = state.scene.camera;

            if (auto v = readFloat("cam.position.input.x")) cam.position.x = *v;
            if (auto v = readFloat("cam.position.input.y")) cam.position.y = *v;
            if (auto v = readFloat("cam.position.input.z")) cam.position.z = *v;

            if (auto v = readFloat("cam.target.input.x"))   cam.target.x   = *v;
            if (auto v = readFloat("cam.target.input.y"))   cam.target.y   = *v;
            if (auto v = readFloat("cam.target.input.z"))   cam.target.z   = *v;

            if (auto v = readFloat("cam.up.input.x"))       cam.up.x       = *v;
            if (auto v = readFloat("cam.up.input.y"))       cam.up.y       = *v;
            if (auto v = readFloat("cam.up.input.z"))       cam.up.z       = *v;

            if (auto v = readFloat("cam.vfov.input"))       cam.vfovDeg    = *v;
            if (auto v = readFloat("cam.aspect.input"))     cam.aspect     = *v;
            if (auto v = readFloat("cam.near.input"))       cam.nearPlane  = *v;
            if (auto v = readFloat("cam.far.input"))        cam.farPlane   = *v;

            state.scene.camera = cam;
        }
        else if (uiState.previousTargetEditor >= 0)
        {
            const int prevIdx = uiState.previousTargetEditor;
        
            auto optSid = sidForIndex(models, prevIdx);
            if (!optSid) { uiState.previousTargetEditor = uiState.targetEditor; return; }
            Sid sid = *optSid;
        
            SceneObject objCopy{};
            if ((size_t)prevIdx < state.scene.objects.size())
                objCopy = state.scene.objects[prevIdx];
        
            Transform         transform = objCopy.transform;
            RotationAnimation anim      = objCopy.animation;
        
            auto rf = [&](const char* key) -> std::optional<float> {
                Clay_String str{false, (int32_t)std::strlen(key), const_cast<char*>(key)};
                Clay_ElementId id = Clay_GetElementIdWithIndex(str, sid);
                return readFloatById(id);
            };
        
            if (auto v = rf("model.translate.input.x")) transform.translate.x = *v;
            if (auto v = rf("model.translate.input.y")) transform.translate.y = *v;
            if (auto v = rf("model.translate.input.z")) transform.translate.z = *v;
        
            if (auto v = rf("model.rotation.input.x"))  transform.rotation.x  = *v;
            if (auto v = rf("model.rotation.input.y"))  transform.rotation.y  = *v;
            if (auto v = rf("model.rotation.input.z"))  transform.rotation.z  = *v;
        
            if (auto v = rf("model.scale.input.x"))     transform.scale.x     = *v;
            if (auto v = rf("model.scale.input.y"))     transform.scale.y     = *v;
            if (auto v = rf("model.scale.input.z"))     transform.scale.z     = *v;
        
            if (auto v = rf("model.rotspeed.input.x"))  anim.rotationSpeedPerAxis.x = *v;
            if (auto v = rf("model.rotspeed.input.y"))  anim.rotationSpeedPerAxis.y = *v;
            if (auto v = rf("model.rotspeed.input.z"))  anim.rotationSpeedPerAxis.z = *v;
        
            if (auto v = rf("model.speedscalar.input")) anim.SpeedScalar = *v;
        
            if ((size_t)prevIdx >= state.scene.objects.size())
                state.scene.objects.resize((size_t)prevIdx + 1);
        
            state.scene.objects[prevIdx].transform = transform;
            state.scene.objects[prevIdx].animation = anim;
        
            uiState.previousTargetEditor = uiState.targetEditor;
        }
    }
}

void UiApp::HandleErrorShowing(Clay_ElementId elementId, Clay_PointerData pointerData)
{
    if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && ((elementId.id == Clay_GetElementId(CLAY_STRING("backgroundError")).id) || (elementId.id == Clay_GetElementId(CLAY_STRING("Empty1")).id)))
    {
        uiState.showError = false;
    }

}

void UiApp::HandleButtonInteraction(Clay_ElementId elementId, Clay_PointerData pointerData) 
{

    if (isInput(elementId))
        wanted = cursorIBeam;
    else if (elementId.id != Clay_GetElementId(CLAY_STRING("background")).id)
        wanted = cursorHand;

    
    if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("background")).id)) {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("Device Selector")).id)) {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};
        uiState.showDevicePicker = true;
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("model loader")).id)) {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};
        auto eId = Clay_GetElementId(CLAY_STRING("input form"));
        if (!isObj(inputs.get(eId).text))
        {
            uiState.errorMsg = "The model path is not pointing to .obj";
            uiState.showError = true;
        }
        else if (uiState.loadingModel)
        {
            uiState.errorMsg = "A model is beeing loaded, please Wait";
            uiState.showError = true;
        }
        else
        {
            fut = std::async(std::launch::async,[path = std::string(inputs.get(eId).text)]()
            {
                return Object(path);
            });
            uiState.loadingModel = true;
        }
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("launch button")).id)) {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};
        if (!allBVHReady())
        {
            uiState.errorMsg  = "Not all Model SBVH's are Built";
            uiState.showError = true;
            return;
        }
        UpdateInput(true);
        finalizeSceneForLaunch();
        state.shouldClose = false;
        if (state.device == VK_NULL_HANDLE)
            state.device = device.getPhysicalDevice();
        exitRun = true;
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("Menu Selector")).id))
    {
        auto eId = Clay_GetElementId(CLAY_STRING("input form"));
        try
        {
            inputs.get(eId).text = pickObjRelativePath();
        }
        catch(const std::exception& e)
        {
            uiState.errorMsg = std::string(e.what());
            uiState.showError = true;
        }
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.baseId == Clay_GetElementId(CLAY_STRING("tab.extra")).baseId))
    {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};

        Sid sid = (Sid)elementId.offset;
        auto it = sidToIndex.find(sid);
        if (it != sidToIndex.end())
            uiState.targetEditor = (int)it->second;

        // int idx = (int)elementId.offset;
        // if (idx >= 0 && idx < (int)models.size())
        // uiState.targetEditor = idx;
        UpdateInput();
        
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && elementId.id == Clay_GetElementId(CLAY_STRING("tab.camera")).id)
    {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};
        uiState.targetEditor = -1;
        UpdateInput();
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME)
    {
        inputs.blurAll();
        focusedInputId = Clay_ElementId{0};
    }
    HandleMultiInput(elementId, pointerData);
}

void UiApp::HandleFloatingShowing(Clay_ElementId elementId, Clay_PointerData pointerData)
{
    Clay_ElementId rowBase = Clay_GetElementId(CLAY_STRING("picker.row"));
    if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("Device Selector")).id))
    {
        uiState.showDevicePicker = true;
    }
    else if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME && (elementId.id == Clay_GetElementId(CLAY_STRING("Selector Exit")).id))
    {
        uiState.showDevicePicker = false;
    }

    if (elementId.baseId == rowBase.baseId) {
        uint32_t idx = elementId.offset;
        wanted = cursorHand;
        
        if (pointerData.state == CLAY_POINTER_DATA_PRESSED_THIS_FRAME)
        {
            if (idx < optionalDeviceNames.size())
            {
                const std::string& name = optionalDeviceNames[idx];
                uiState.selectedDeviceName = name;
                uiState.showDevicePicker   = false;
                state.device = optionalDevices[idx];
            }
        }
    }
}

void UiApp::HandleMultiInput(Clay_ElementId elementId, Clay_PointerData pointerData)
{
    if (!PressedThis(pointerData))
        return;


    if (isInput(elementId))
    {
        wanted = cursorIBeam;
        inputs.focus(elementId);
        if (auto* s = inputs.focused())
        {
            s->caret = s->text.size();
            s->blinkStart = glfwGetTime();
        }
        focusedInputId = elementId;
        return;
    }
    else if (isModelButton(elementId))
    {
        if (matchesBase(elementId, Clay_GetElementId(CLAY_STRING("model.instance.add"))))
        {
            if (uiState.targetEditor == -1)
                return;
            auto ptr = models[uiState.targetEditor].object;
            models.push_back(ModelTab{ models[uiState.targetEditor].name, ptr, nextSid++});
            state.scene.objects.emplace_back(SceneObject{});
            rebuildSidIndexMap();
        }
        else if (matchesBase(elementId, Clay_GetElementId(CLAY_STRING("model.instance.del"))))
        {
            Sid deadSid = models[uiState.targetEditor].stableID;
            models.erase(models.begin() + uiState.targetEditor);
            state.scene.objects.erase(state.scene.objects.begin() + uiState.targetEditor);
            uiState.targetEditor--;
            rebuildSidIndexMap();
        }
        else if (matchesBase(elementId, Clay_GetElementId(CLAY_STRING("model.build"))))
        {
            Sid sid = elementId.offset;
            auto optIdx = indexFromTabSid(sid, models);
            if (!optIdx) return;
            size_t idx = *optIdx;
        
            const std::string& name = models[idx].name;
        
            if (findMeshIndexByName(state.scene, name) >= 0) {
                uiState.errorMsg = "BVH already built for \"" + name + "\".";
                uiState.showError = true;
                return;
            }
        
            if (bvhInFlight.count(name)) {
                uiState.errorMsg = "BVH is already building for \"" + name + "\". Please wait.";
                uiState.showError = true;
                return;
            }
        
            auto objSP = models[idx].object;
            bvhInFlight.insert(name);
        
            pendingBVH.push_back(PendingBVH{
                name,
                idx,
                std::async(std::launch::async, [name, objSP]() -> ObjectMeshData {
                    ObjectMeshData out;
                    out.name  = name;
                    out.bottomLevelAccelerationStructure = objSP->buildSplitBoundingVolumeHierarchy();
                    out.perMeshMaterials = objSP->buildMaterialGPU();
                    out.textures = objSP->getTextures();
                    return out;
                })
            });
        
        }
    }

}

void UiApp::pollBVHBuilders()
{
    using namespace std::chrono_literals;

    for (size_t i = 0; i < pendingBVH.size(); ) {
        auto& p = pendingBVH[i];
        if (p.fut.wait_for(0ms) == std::future_status::ready) {
            try {
                ObjectMeshData data = p.fut.get();

                if (findMeshIndexByName(state.scene, data.name) < 0)
                    state.scene.meshes.push_back(std::move(data));

                bvhInFlight.erase(p.name);
            }
            catch (const std::exception& e)
            {
                bvhInFlight.erase(p.name);
                uiState.errorMsg = std::string("BVH build failed for \"") + p.name + "\": " + e.what();
                uiState.showError = true;
            }
            pendingBVH[i] = std::move(pendingBVH.back());
            pendingBVH.pop_back();
        }
        else
            i++;
    }
}


void UiApp::buildUi() 
{
    
    Clay_SetLayoutDimensions((Clay_Dimensions) {static_cast<float>(window.getWidth()), static_cast<float>(window.getHeight())});
    Clay_SetMeasureTextFunction(MeasureTextFromAtlas, text.get());
    std::string name = !uiState.selectedDeviceName.empty() ? uiState.selectedDeviceName  : device.getName();
    clayFrameStrings.clear();
    wanted = cursorArrow;
    bool isReady = allBVHReady();
    if (fut.valid())
    {
        using namespace std::chrono_literals;
        auto s = fut.wait_for(0ms);
        if (s == std::future_status::ready)
        {
            uiState.loadingModel = false;
            try
            {
                Object loaded = fut.get();
                auto obj = std::make_shared<Object>(std::move(loaded));
                models.push_back(ModelTab{obj->getFileName(), obj, nextSid++});
                state.scene.objects.emplace_back(SceneObject{});
                rebuildSidIndexMap();
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                std::string errMsg(e.what());
                uiState.errorMsg = errMsg;
                uiState.showError = true;
            }
        }
    }
    pollBVHBuilders();
    
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
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)}, 
                    .padding = CLAY_PADDING_ALL(16),
                    .childAlignment = { .x = CLAY_ALIGN_X_LEFT, .y = CLAY_ALIGN_Y_CENTER },
                    .layoutDirection = CLAY_LEFT_TO_RIGHT
                }, 
                .backgroundColor = {51, 30, 108, 170}, 
                .cornerRadius = CLAY_CORNER_RADIUS(20),
                .clip = {.horizontal = true, .vertical = true}
                }){
                    Clay_ElementId launchID = Clay_GetElementId(CLAY_STRING("launch button"));
                    bool hovered = Clay_PointerOver(launchID);
            CLAY_TEXT(CLAY_STRING("Welcome To Scoop"), 
                CLAY_TEXT_CONFIG({ 
                    .textColor = {255, 243, 232, 255}, 
                    .fontSize = 128, 
                    .letterSpacing = 1, 
                    .wrapMode = CLAY_TEXT_WRAP_NONE, 
                    .textAlignment = CLAY_TEXT_ALIGN_CENTER
                }));
                CLAY({.id = CLAY_ID("emptysizer"), .layout = {.sizing = {CLAY_SIZING_GROW(0),CLAY_SIZING_GROW(0)}}});
            CLAY({ .id = launchID,
                .layout = {
                    .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                    .padding = CLAY_PADDING_ALL(16),
                    .childGap = 16,
                    .layoutDirection = CLAY_LEFT_TO_RIGHT
                },
                .backgroundColor = hovered ? (Clay_Color){139, 85, 199,255} : isReady ? (Clay_Color){65, 9, 114, 200} : (Clay_Color){55, 9, 104, 200},
                .cornerRadius = CLAY_CORNER_RADIUS(20),
                .clip = {.horizontal = true, .vertical = true}
            }){
                Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                CLAY_TEXT(CLAY_STRING("Launch"), 
                CLAY_TEXT_CONFIG({
                    .textColor = isReady ? (Clay_Color){255, 243, 232, 255} : (Clay_Color){155, 143, 132, 155}, 
                    .fontSize = 64, 
                    .letterSpacing = 1, 
                    .wrapMode = CLAY_TEXT_WRAP_NONE, 
                    .textAlignment = CLAY_TEXT_ALIGN_CENTER
                }));
            }
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
                .fontSize = 32, 
                .letterSpacing = 1, 
                .wrapMode = CLAY_TEXT_WRAP_NONE, 
                .textAlignment = CLAY_TEXT_ALIGN_CENTER
            }));
            CLAY({.layout = {.sizing = {CLAY_SIZING_GROW(0)}}});
            CLAY({ .id = CLAY_ID("Device Selector"),
                .layout = {
                    .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                    .padding = CLAY_PADDING_ALL(10),
                },
                .backgroundColor = {65, 9, 114, 200},
                .cornerRadius = CLAY_CORNER_RADIUS(35),
                .clip = {.horizontal = true, .vertical = true}
            }){
                Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                CLAY_TEXT(ClayFromStable(clayFrameStrings ,name), 
                CLAY_TEXT_CONFIG({
                    .textColor = {255, 243, 232, 255}, 
                    .fontSize = 32, 
                    .letterSpacing = 1, 
                    .wrapMode = CLAY_TEXT_WRAP_NONE, 
                    .textAlignment = CLAY_TEXT_ALIGN_CENTER
                }));
            }
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
                CLAY({ .id = CLAY_ID("Input Wraper"),
                    .layout = {
                            .sizing = {CLAY_SIZING_GROW(0, 1650), CLAY_SIZING_GROW(0)},
                            .padding = CLAY_PADDING_ALL(10),
                            .childGap = 9,
                            .layoutDirection = CLAY_LEFT_TO_RIGHT
                        },
                        .backgroundColor = {179, 170, 162, 170}, 
                        .cornerRadius = CLAY_CORNER_RADIUS(20),
                }){
                    CLAY({ .id = CLAY_ID("input form"),
                        .layout = {
                            .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)},
                            .padding = CLAY_PADDING_ALL(10)
                        },
                        .backgroundColor = {0,0,0,0}, 
                        .cornerRadius = CLAY_CORNER_RADIUS(20),
                        .clip = {.horizontal = true, .vertical = true}
                    }){
                        Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                        auto eId = Clay_GetElementId(CLAY_STRING("input form"));
                        CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(eId).text),
                        CLAY_TEXT_CONFIG({
                            .textColor = {255, 243, 232, 255},
                            .fontSize  = 32,
                            .letterSpacing = 1,
                            .wrapMode  = CLAY_TEXT_WRAP_NONE,
                            .textAlignment = CLAY_TEXT_ALIGN_LEFT
                        }));
                    }
                    CLAY({ .id = CLAY_ID("Menu Selector"),
                        .layout = {
                            .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0)},
                            .padding = CLAY_PADDING_ALL(16),
                            
                        },
                        .backgroundColor = {109, 100, 92, 200},
                        .cornerRadius = CLAY_CORNER_RADIUS(10),
                        .clip = {.horizontal = true, .vertical = true}
                    }){
                        Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                        CLAY_TEXT(CLAY_STRING("Select Object"),
                        CLAY_TEXT_CONFIG({
                            .textColor = {255, 243, 232, 255}, 
                            .fontSize = 32, 
                            .letterSpacing = 1, 
                            .wrapMode = CLAY_TEXT_WRAP_NONE, 
                            .textAlignment = CLAY_TEXT_ALIGN_CENTER
                        }));
                    }
                }
                CLAY({ .id = CLAY_ID("model loader"),
                        .layout = {
                            .sizing = {CLAY_SIZING_FIT(0), CLAY_SIZING_GROW(0)},
                            .padding = CLAY_PADDING_ALL(10),
                            .childAlignment = {
                              .x = CLAY_ALIGN_X_CENTER,
                              .y = CLAY_ALIGN_Y_CENTER
                            },
                        },
                        .backgroundColor = uiState.loadingModel ? (Clay_Color){65, 9, 114, 50} : (Clay_Color){65, 9, 114, 200},
                        .cornerRadius = CLAY_CORNER_RADIUS(35),
                        .clip = {.horizontal = true, .vertical = true}
                }){
                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                    CLAY_TEXT(CLAY_STRING("Load Model"), 
                    CLAY_TEXT_CONFIG({
                        .textColor = uiState.loadingModel ? (Clay_Color){255, 243, 232, 80} : (Clay_Color){255, 243, 232, 255}, 
                        .fontSize = 32, 
                        .letterSpacing = 1, 
                        .wrapMode = CLAY_TEXT_WRAP_NONE, 
                        .textAlignment = CLAY_TEXT_ALIGN_CENTER
                    }));
                }
            }
        }
        CLAY({ .id = CLAY_ID("tech panel"),
            .layout = {
                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                .padding = CLAY_PADDING_ALL(16),
                .childGap = 12,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .backgroundColor = {51, 30, 108, 170},
            .cornerRadius    = CLAY_CORNER_RADIUS(20),
            .clip = { .horizontal = true, .vertical = true }
        }) {
            CLAY({
                .id = CLAY_ID("tech navbar"),
                .layout = {
                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIXED(56) },
                    .padding = CLAY_PADDING_ALL(8),
                    .childGap = 8,
                    .layoutDirection = CLAY_LEFT_TO_RIGHT
                },
                .backgroundColor = {38,29,65,200},
                .cornerRadius    = CLAY_CORNER_RADIUS(18),
            }) {
                {
                    Clay_ElementId tabId = Clay_GetElementId(CLAY_STRING("tab.camera"));
                    bool hovered = Clay_PointerOver(tabId);
                
                    CLAY({
                        .id = tabId,
                        .layout = {
                            .sizing = { CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0) },
                            .padding = CLAY_PADDING_ALL(10),
                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                        },
                        .backgroundColor = hovered ? (Clay_Color){139, 85, 199,255} : uiState.targetEditor == -1 ? (Clay_Color){109, 55, 169,255} : (Clay_Color){65, 9, 114, 200},
                        .cornerRadius = CLAY_CORNER_RADIUS(10),
                        .clip = { .horizontal = true, .vertical = true }
                    }) {
                        Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                        CLAY_TEXT(CLAY_STRING("Camera"),
                            CLAY_TEXT_CONFIG({
                                .textColor = {255, 243, 232, 255},
                                .fontSize = 32,
                                .wrapMode = CLAY_TEXT_WRAP_NONE,
                                .textAlignment = CLAY_TEXT_ALIGN_CENTER
                            })
                        );
                    }
                }
            
                for (int i = 0; i < (int)models.size(); i++)
                {
                    Sid sid = models[i].stableID;
                    Clay_ElementId tabId = Clay_GetElementIdWithIndex(CLAY_STRING("tab.extra"), sid);
                    bool hovered = Clay_PointerOver(tabId);
                
                    CLAY({
                        .id = tabId,
                        .layout = {
                            .sizing = { CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0) },
                            .padding = CLAY_PADDING_ALL(10),
                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                        },
                        .backgroundColor = hovered ? (Clay_Color){139, 85, 199,255} : uiState.targetEditor == i ? (Clay_Color){109, 55, 169,255} : (Clay_Color){65, 9, 114, 200},
                        .cornerRadius = CLAY_CORNER_RADIUS(10),
                        .clip = { .horizontal = true, .vertical = true }
                    }) {
                        Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                        CLAY_TEXT(ClayFromStable(clayFrameStrings, models[i].name),
                            CLAY_TEXT_CONFIG({
                                .textColor = {255, 243, 232, 255},
                                .fontSize = 32,
                                .wrapMode = CLAY_TEXT_WRAP_NONE,
                                .textAlignment = CLAY_TEXT_ALIGN_CENTER
                            })
                        );
                    }
                }
            }
            CLAY({ .id = CLAY_ID("tech content"),
                .layout = {
                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                    .padding = CLAY_PADDING_ALL(12),
                    .childGap = 8,
                    .layoutDirection = CLAY_TOP_TO_BOTTOM
                },
                .backgroundColor = {38,29,65,200},
                .cornerRadius    = CLAY_CORNER_RADIUS(10),
            }) {
                if (uiState.targetEditor == -1)
                {
                    CLAY({
                        .id = CLAY_ID("cam.panel"),
                        .layout = {
                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                            .childGap = 12,
                            .layoutDirection = CLAY_TOP_TO_BOTTOM
                        }
                    }) {
                        CLAY({
                            .id = CLAY_ID("cam.row1"),
                            .layout = {
                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                                .childGap = 12,
                                .layoutDirection = CLAY_LEFT_TO_RIGHT
                            }
                        }) {
                            CLAY({
                                .id = CLAY_ID("cam.position.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 10,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Position"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.position.row"),
                                    .layout = {
                                        .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                        .childGap = 10,
                                        .layoutDirection = CLAY_LEFT_TO_RIGHT
                                    }
                                }) {
                                    CLAY({
                                        .id = CLAY_ID("cam.position.stack.x"),
                                        .layout = {
                                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                            .childGap = 6,
                                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                            .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                        }
                                    }) {
                                        CLAY_TEXT(CLAY_STRING("X"),
                                            CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                        CLAY({
                                            .id = CLAY_ID("cam.position.input.x"),
                                            .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                            .backgroundColor = (Clay_Color){230,224,217,255},
                                            .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                            .clip = { .horizontal = true, .vertical = true }
                                        }) {
                                            Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                            CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.position.input.x"))).text),
                                                CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_LEFT }));
                                        }
                                    }
                                    CLAY({
                                        .id = CLAY_ID("cam.position.stack.y"),
                                        .layout = {
                                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                            .childGap = 6,
                                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                            .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                        }
                                    }) {
                                        CLAY_TEXT(CLAY_STRING("Y"),
                                            CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                        CLAY({
                                            .id = CLAY_ID("cam.position.input.y"),
                                            .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                            .backgroundColor = (Clay_Color){230,224,217,255},
                                            .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                            .clip = { .horizontal = true, .vertical = true }
                                        }) {
                                            Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                            CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.position.input.y"))).text),
                                                CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                        }
                                    }
                                    CLAY({
                                        .id = CLAY_ID("cam.position.stack.z"),
                                        .layout = {
                                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                            .childGap = 6,
                                            .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                            .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                        }
                                    }) {
                                        CLAY_TEXT(CLAY_STRING("Z"),
                                            CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                        CLAY({
                                            .id = CLAY_ID("cam.position.input.z"),
                                            .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                            .backgroundColor = (Clay_Color){230,224,217,255},
                                            .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                            .clip = { .horizontal = true, .vertical = true }
                                        }) {
                                            Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                            CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.position.input.z"))).text),
                                                CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                        }
                                    }
                                }
                            }
                            CLAY({
                                .id = CLAY_ID("cam.target.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 10,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Target"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.target.row"),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT }
                                }) {
                                    auto mkVecInput = [&](const char* id, const char* label) {
                                        CLAY({
                                            .id = Clay_GetElementId(ClayFromStable(clayFrameStrings, std::string("cam.target.stack.") + label)),
                                            .layout = {
                                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                                .childGap = 6,
                                                .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                                .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                            }
                                        }) {
                                            CLAY_TEXT(ClayFromStable(clayFrameStrings, label),
                                                CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                            CLAY({
                                                .id = Clay_GetElementId(ClayFromStable(clayFrameStrings, id)),
                                                .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                                .backgroundColor = (Clay_Color){230,224,217,255},
                                                .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                                .clip = { .horizontal = true, .vertical = true }
                                            }) {
                                                Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                                CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(ClayFromStable(clayFrameStrings, id))).text),
                                                    CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                            }
                                        };
                                    };
                                
                                    mkVecInput("cam.target.input.x", "X");
                                    mkVecInput("cam.target.input.y", "Y");
                                    mkVecInput("cam.target.input.z", "Z");
                                }
                            }
                            CLAY({
                                .id = CLAY_ID("cam.up.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 10,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Up"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.up.row"),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT }
                                }) {
                                    auto mkVecInput = [&](const char* id, const char* label) {
                                        CLAY({
                                            .id = Clay_GetElementId(ClayFromStable(clayFrameStrings, std::string("cam.up.stack.") + label)),
                                            .layout = {
                                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                                .childGap = 6,
                                                .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                                .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                            }
                                        }) {
                                            CLAY_TEXT(ClayFromStable(clayFrameStrings, label),
                                                CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                            CLAY({
                                                .id = Clay_GetElementId(ClayFromStable(clayFrameStrings, id)),
                                                .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                                .backgroundColor = (Clay_Color){230,224,217,255},
                                                .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                                .clip = { .horizontal = true, .vertical = true }
                                            }) {
                                                Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                                CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(ClayFromStable(clayFrameStrings, id))).text),
                                                    CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                            }
                                        };
                                    };
                                    mkVecInput("cam.up.input.x", "X");
                                    mkVecInput("cam.up.input.y", "Y");
                                    mkVecInput("cam.up.input.z", "Z");
                                }
                            }
                        }
                        CLAY({
                            .id = CLAY_ID("cam.row2"),
                            .layout = {
                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                                .childGap = 12,
                                .layoutDirection = CLAY_LEFT_TO_RIGHT
                            }
                        }) {
                            CLAY({
                                .id = CLAY_ID("cam.vfov.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 8,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = {51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("VFOV (deg)"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.vfov.input"),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.vfov.input"))).text),
                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                }
                            }
                            CLAY({
                                .id = CLAY_ID("cam.aspect.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 8,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = {51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Aspect"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.aspect.input"),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.aspect.input"))).text),
                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                }
                            }
                            CLAY({
                                .id = CLAY_ID("cam.near.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 8,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = {51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Near"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.near.input"),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.near.input"))).text),
                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                }
                            }
                            CLAY({
                                .id = CLAY_ID("cam.far.card"),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 8,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = {51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Far"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = CLAY_ID("cam.far.input"),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementId(CLAY_STRING("cam.far.input"))).text),
                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                }
                            }
                        }
                    }
                }
                else
                {
                    Sid idx = models[uiState.targetEditor].stableID;
                
                    CLAY({
                        .id = CLAY_ID("model.panel"),
                        .layout = {
                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                            .childGap = 12,
                            .layoutDirection = CLAY_TOP_TO_BOTTOM
                        }
                    }) {
                        CLAY({
                            .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.row.transform"), idx),
                            .layout = {
                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                .childGap = 12,
                                .layoutDirection = CLAY_LEFT_TO_RIGHT
                            }
                        }) {
                            auto mkVec3Card = [&](const char* title,
                                                  Clay_ElementId idX,
                                                  Clay_ElementId idY,
                                                  Clay_ElementId idZ)
                            {
                                CLAY({
                                    .id = Clay_GetElementIdWithIndex(ClayFromStable(clayFrameStrings, std::string("model.card.") + title), idx),
                                    .layout = {
                                        .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                        .padding = CLAY_PADDING_ALL(12),
                                        .childGap = 10,
                                        .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                        .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                    },
                                    .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, title),
                                        CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                
                                    // Unique row id per (title, idx)
                                    CLAY({
                                        .id = Clay_GetElementIdWithIndex(ClayFromStable(clayFrameStrings, std::string("model.vecrow.") + title), idx),
                                        .layout = {
                                            .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                            .childGap = 10,
                                            .layoutDirection = CLAY_LEFT_TO_RIGHT
                                        }
                                    }) {
                                        auto oneAxis = [&](const char* axisLabel, Clay_ElementId inputId){
                                            CLAY({
                                                .id = Clay_GetElementIdWithIndex(ClayFromStable(clayFrameStrings, std::string(title) + ".stack." + axisLabel), idx),
                                                .layout = {
                                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                                    .childGap = 6,
                                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                                }
                                            }) {
                                                CLAY_TEXT(ClayFromStable(clayFrameStrings, axisLabel),
                                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                                CLAY({
                                                    .id = inputId,
                                                    .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                                    .clip = { .horizontal = true, .vertical = true }
                                                }) {
                                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(inputId).text),
                                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                                }
                                            };
                                        };
                                    
                                        oneAxis("X", idX);
                                        oneAxis("Y", idY);
                                        oneAxis("Z", idZ);
                                    }
                                };
                            };
                        
                            auto mkRotCard = [&](){
                                CLAY({
                                    .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.rotation.card"), idx),
                                    .layout = {
                                        .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                        .padding = CLAY_PADDING_ALL(12),
                                        .childGap = 10,
                                        .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                        .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                    },
                                    .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    CLAY_TEXT(CLAY_STRING("Rotation (deg)"),
                                        CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                
                                    CLAY({
                                        .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.rotation.row"), idx),
                                        .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT }
                                    }) {
                                        auto axis = [&](const char* axisLabel, Clay_ElementId inputId){
                                            CLAY({
                                                .id = Clay_GetElementIdWithIndex(ClayFromStable(clayFrameStrings, std::string("rot.stack.") + axisLabel), idx),
                                                .layout = {
                                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                                    .childGap = 6,
                                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                                }
                                            }) {
                                                CLAY_TEXT(ClayFromStable(clayFrameStrings, axisLabel),
                                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                                CLAY({
                                                    .id = inputId,
                                                    .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                                    .clip = { .horizontal = true, .vertical = true }
                                                }) {
                                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(inputId).text),
                                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                                }
                                            };
                                        };
                                    
                                        axis("X", Clay_GetElementIdWithIndex(CLAY_STRING("model.rotation.input.x"), idx));
                                        axis("Y", Clay_GetElementIdWithIndex(CLAY_STRING("model.rotation.input.y"), idx));
                                        axis("Z", Clay_GetElementIdWithIndex(CLAY_STRING("model.rotation.input.z"), idx));
                                    }
                                };
                            };
                        
                            auto mkScaleCard = [&](){
                                CLAY({
                                    .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.scale.card"), idx),
                                    .layout = {
                                        .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                        .padding = CLAY_PADDING_ALL(12),
                                        .childGap = 10,
                                        .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                        .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                    },
                                    .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    CLAY_TEXT(CLAY_STRING("Scale"),
                                        CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                
                                    CLAY({
                                        .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.scale.row"), idx),
                                        .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT }
                                    }) {
                                        auto axis = [&](const char* axisLabel, Clay_ElementId inputId){
                                            CLAY({
                                                .id = Clay_GetElementIdWithIndex(ClayFromStable(clayFrameStrings, std::string("scale.stack.") + axisLabel), idx),
                                                .layout = {
                                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                                    .childGap = 6,
                                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                                }
                                            }) {
                                                CLAY_TEXT(ClayFromStable(clayFrameStrings, axisLabel),
                                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                                CLAY({
                                                    .id = inputId,
                                                    .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                                    .clip = { .horizontal = true, .vertical = true }
                                                }) {
                                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(inputId).text),
                                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                                }
                                            };
                                        };
                                    
                                        axis("X", Clay_GetElementIdWithIndex(CLAY_STRING("model.scale.input.x"), idx));
                                        axis("Y", Clay_GetElementIdWithIndex(CLAY_STRING("model.scale.input.y"), idx));
                                        axis("Z", Clay_GetElementIdWithIndex(CLAY_STRING("model.scale.input.z"), idx));
                                    }
                                };
                            };
                        
                            mkVec3Card("Translate",
                                Clay_GetElementIdWithIndex(CLAY_STRING("model.translate.input.x"), idx),
                                Clay_GetElementIdWithIndex(CLAY_STRING("model.translate.input.y"), idx),
                                Clay_GetElementIdWithIndex(CLAY_STRING("model.translate.input.z"), idx));
                            mkRotCard();
                            mkScaleCard();
                        }
                    
                        CLAY({
                            .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.row.anim"), idx),
                            .layout = {
                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                .childGap = 12,
                                .layoutDirection = CLAY_LEFT_TO_RIGHT
                            }
                        }) {
                            CLAY({
                                .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.rotspeed.card"), idx),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 10,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Rotation Speed / Axis"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                            
                                CLAY({
                                    .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.rotspeed.row"), idx),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT }
                                }) {
                                    auto axis = [&](const char* axisLabel, Clay_ElementId inputId){
                                        CLAY({
                                            .id = Clay_GetElementIdWithIndex(ClayFromStable(clayFrameStrings, std::string("rotspeed.stack.") + axisLabel), idx),
                                            .layout = {
                                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                                .childGap = 6,
                                                .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                                .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                            }
                                        }) {
                                            CLAY_TEXT(ClayFromStable(clayFrameStrings, axisLabel),
                                                CLAY_TEXT_CONFIG({ .textColor = {255,243,232,220}, .fontSize = 24, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                            CLAY({
                                                .id = inputId,
                                                .layout = { .sizing = { CLAY_SIZING_FIXED(160), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                                .backgroundColor = (Clay_Color){230,224,217,255},
                                                .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                                .clip = { .horizontal = true, .vertical = true }
                                            }) {
                                                Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                                CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(inputId).text),
                                                    CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                            }
                                        };
                                    };
                                
                                    axis("X", Clay_GetElementIdWithIndex(CLAY_STRING("model.rotspeed.input.x"), idx));
                                    axis("Y", Clay_GetElementIdWithIndex(CLAY_STRING("model.rotspeed.input.y"), idx));
                                    axis("Z", Clay_GetElementIdWithIndex(CLAY_STRING("model.rotspeed.input.z"), idx));
                                }
                            }
                        
                            CLAY({
                                .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.speed.card"), idx),
                                .layout = {
                                    .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
                                    .padding = CLAY_PADDING_ALL(12),
                                    .childGap = 8,
                                    .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_TOP },
                                    .layoutDirection = CLAY_TOP_TO_BOTTOM,
                                },
                                .backgroundColor = (Clay_Color){51, 30, 108, 170},
                                .cornerRadius    = CLAY_CORNER_RADIUS(16),
                                .clip = { .horizontal = true, .vertical = true }
                            }) {
                                CLAY_TEXT(CLAY_STRING("Speed Multiplier"),
                                    CLAY_TEXT_CONFIG({ .textColor = {255,243,232,255}, .fontSize = 32, .wrapMode = CLAY_TEXT_WRAP_NONE, .textAlignment = CLAY_TEXT_ALIGN_CENTER }));
                                CLAY({
                                    .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.speedscalar.input"), idx),
                                    .layout = { .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) }, .padding = CLAY_PADDING_ALL(10) },
                                    .backgroundColor = (Clay_Color){230,224,217,255},
                                    .cornerRadius    = CLAY_CORNER_RADIUS(10),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                    inputs.get(Clay_GetElementIdWithIndex(CLAY_STRING("model.speedscalar.input"), idx)).fontPx = 64;
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, inputs.get(Clay_GetElementIdWithIndex(CLAY_STRING("model.speedscalar.input"), idx)).text),
                                        CLAY_TEXT_CONFIG({ .textColor = {40,35,50,255}, .fontSize = 64, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                                }
                            }
                        }
                    
                        CLAY({
                            .id = Clay_GetElementIdWithIndex(CLAY_STRING("model.row.actions"), idx),
                            .layout = {
                                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                                .childGap = 8,
                                .layoutDirection = CLAY_LEFT_TO_RIGHT
                            }
                        }) {
                            auto mkBtn = [&](const char* label, Clay_ElementId id){
                                bool hovered = Clay_PointerOver(id);
                                CLAY({
                                    .id = id,
                                    .layout = {
                                        .sizing = { CLAY_SIZING_FIT(0), CLAY_SIZING_FIT(0) },
                                        .padding = CLAY_PADDING_ALL(10),
                                        .childAlignment = { .x = CLAY_ALIGN_X_CENTER, .y = CLAY_ALIGN_Y_CENTER }
                                    },
                                    .backgroundColor = hovered ? (Clay_Color){139, 85, 199,255} : (Clay_Color){65, 9, 114, 200},
                                    .cornerRadius = CLAY_CORNER_RADIUS(10),
                                    .clip = { .horizontal = true, .vertical = true }
                                }) {
                                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                                    CLAY_TEXT(ClayFromStable(clayFrameStrings, label),
                                        CLAY_TEXT_CONFIG({
                                            .textColor = {255, 243, 232, 255},
                                            .fontSize = 32,
                                            .wrapMode = CLAY_TEXT_WRAP_NONE,
                                            .textAlignment = CLAY_TEXT_ALIGN_CENTER
                                        }));
                                }
                            };
                        
                            mkBtn("Build SBVH & Materials", Clay_GetElementIdWithIndex(CLAY_STRING("model.build"), idx));
                            mkBtn("Add Instance",          Clay_GetElementIdWithIndex(CLAY_STRING("model.instance.add"), idx));
                            mkBtn("Delete Instance",       Clay_GetElementIdWithIndex(CLAY_STRING("model.instance.del"), idx));
                        }
                    }
                }

            }
        }
    }

    if (uiState.showDevicePicker)
    {
        if (optionalDeviceNames.empty() || optionalDevices.empty())
        {
            std::vector<VkPhysicalDevice> options = device.getOptionalDevices();
            std::vector<std::string> names;
            names.reserve(options.size());
            for (auto& dev : options)
            {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(dev, &properties);
                names.push_back(std::string(properties.deviceName));
            }
            optionalDevices = options;
            optionalDeviceNames = names;
        }
        CLAY({ .id = CLAY_ID("Selector Exit"),
            .layout = {
                .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0) },
            },
            .backgroundColor = {0,0,0,0},
            .floating = {
                .zIndex = 9998,
                .pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_CAPTURE,
                .attachTo = CLAY_ATTACH_TO_ROOT,
            }
        }) {Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));}
        CLAY({ .id = CLAY_ID("Selector Menu"),
            .layout = {
                .sizing = { CLAY_SIZING_FIXED(360), CLAY_SIZING_FIT(0) },
                .padding = CLAY_PADDING_ALL(10),
                .childGap = 8,
                .layoutDirection = CLAY_TOP_TO_BOTTOM
            },
            .backgroundColor = {51, 30, 108, 170},
            .cornerRadius = CLAY_CORNER_RADIUS(12),
            .floating = {
                .offset = { 0, 8 },
                .parentId = Clay_GetElementId(CLAY_STRING("Device Selector")).id,
                .zIndex = 10000,
                .attachPoints = { .element = CLAY_ATTACH_POINT_RIGHT_CENTER, .parent = CLAY_ATTACH_POINT_LEFT_CENTER },
                .pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_CAPTURE,
                .attachTo = CLAY_ATTACH_TO_ELEMENT_WITH_ID,
                .clipTo  = CLAY_CLIP_TO_ATTACHED_PARENT
            },
            .clip = { .horizontal = true, .vertical = true }
        }) {
            CLAY_TEXT(CLAY_STRING("Select a device"), CLAY_TEXT_CONFIG({ .textColor = {255, 243, 232, 255}, .fontSize = 16 }));

            for (int i = 0; i < (int)optionalDeviceNames.size(); i++)
            {
                Clay_ElementId rowId = Clay_GetElementIdWithIndex(CLAY_STRING("picker.row"), (uint32_t)i);
                bool hovered = Clay_PointerOver(rowId);

                CLAY({
                    .id = rowId,
                    .layout = {
                        .sizing = { CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0) },
                        .padding = { .left=12,.right=12,.top=8,.bottom=8 },
                        .childAlignment = { .x = CLAY_ALIGN_X_LEFT, .y = CLAY_ALIGN_Y_CENTER },
                        .layoutDirection = CLAY_LEFT_TO_RIGHT
                    },
                    .backgroundColor = hovered ? (Clay_Color){139, 85, 199,255} : (Clay_Color){65, 9, 114, 200},
                    .cornerRadius = CLAY_CORNER_RADIUS(8),
                    .clip = { .horizontal = true, .vertical = true }
                }) {
                    Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
                    CLAY_TEXT(ClayFromStable(clayFrameStrings, optionalDeviceNames[i]),
                              CLAY_TEXT_CONFIG({ .textColor = {255, 243, 232, 255}, .fontSize = 16, .wrapMode = CLAY_TEXT_WRAP_NONE }));
                }
            }
        }
    }

    if (uiState.showError)
    {
        CLAY({ .id = CLAY_ID("backgroundError"), 
            .layout = { 
                .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)}, 
                .padding = {.left = 30, .right = 30, .top = 30, .bottom = 30}, 
                .childGap = 16,
                .layoutDirection = CLAY_TOP_TO_BOTTOM,
            },
            .backgroundColor = {0, 0, 0, 0},
            .floating = {
                .zIndex = 10000,
                .pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_CAPTURE,
                .attachTo = CLAY_ATTACH_TO_ROOT,
            },
        }) {
            Clay_OnHover(&UiApp::hoverBridge, reinterpret_cast<intptr_t>(this));
            CLAY({ .id = CLAY_ID("Empty1"),
                .layout = {
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_GROW(0)}
                },
            });
            CLAY({ .id = CLAY_ID("ErrorTextBox"),
                .layout = {
                    .sizing = {CLAY_SIZING_GROW(0), CLAY_SIZING_FIT(0)},
                    .padding =  CLAY_PADDING_ALL(8),  
                },
                .backgroundColor = {255, 243, 232, 255},
                .cornerRadius = CLAY_CORNER_RADIUS(35),
                .clip = {.horizontal = true, .vertical = true}
            }){
                CLAY_TEXT(ClayFromStable(clayFrameStrings , uiState.errorMsg), 
                CLAY_TEXT_CONFIG({
                    .textColor = {255, 21, 21, 255}, 
                    .fontSize = 32, 
                    .letterSpacing = 1, 
                    .wrapMode = CLAY_TEXT_WRAP_NONE, 
                    .textAlignment = CLAY_TEXT_ALIGN_CENTER
                }));
            }
        }
    }

    Clay_RenderCommandArray renderCommands = Clay_EndLayout();

    focusedInputRect = Clay_BoundingBox{0,0,0,0};
    if (focusedInputId.id != 0)
    {
        Clay_ElementData ed = Clay_GetElementData(focusedInputId);
        if (ed.found)
            focusedInputRect = ed.boundingBox;
    }

    {
        double mx, my;
        window.getCursorPos(mx, my);

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

        auto C = [](uint8_t u8){ return u8 / 255.0f; };
        glm::vec4 srgb = { C(rd.backgroundColor.r), C(rd.backgroundColor.g),
                           C(rd.backgroundColor.b), C(rd.backgroundColor.a) };
        it.instance.color = srgbToLinear(srgb); 

        it.instance.radius = { rd.cornerRadius.topLeft,  rd.cornerRadius.topRight,
                           rd.cornerRadius.bottomRight, rd.cornerRadius.bottomLeft };

        it.zIndex = rc.zIndex;
        it.seq = i;
        items.push_back(it);
    }

    std::stable_sort(items.begin(), items.end(),
        [](const RectangleItem& a, const RectangleItem& b){
            if (a.zIndex != b.zIndex) return a.zIndex < b.zIndex;
            return a.seq < b.seq;
        });

    rects.resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) rects[i] = items[i].instance;

    VkExtent2D fb = swapChain->getSwapChainExtent();
    std::vector<VkRect2D> scissorStack;
    scissorStack.push_back(VkRect2D{{0,0},{fb.width, fb.height}});

    for (uint32_t i = 0; i < renderCommands.length; ++i) {
        const Clay_RenderCommand& rc = renderCommands.internalArray[i];

        if (rc.commandType == CLAY_RENDER_COMMAND_TYPE_SCISSOR_START) {
            VkRect2D clipHere = bboxToScissor(rc.boundingBox, fb, 1);

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
        r.scissor = scissorStack.back();
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