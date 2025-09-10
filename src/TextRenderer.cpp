#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include "TextRenderer.hpp"
#include "Device.hpp"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>

// ---- ctor / dtor ------------------------------------------------------------

TextRenderer::TextRenderer(Device& dev, VkCommandPool pool, VkQueue queue, const std::string& ttfPath)
: device(dev), cmdPool(pool), graphicsQueue(queue)
{
    buildAtlasesCPU(ttfPath);

    createDescriptorPoolAndLayout();

    for (auto& A : atlases) uploadAtlas(A);
}

TextRenderer::~TextRenderer() {
    VkDevice dev = device.device();

    if (descPool) { vkDestroyDescriptorPool(dev, descPool, nullptr); descPool = VK_NULL_HANDLE; }
    if (descSetLayout) { vkDestroyDescriptorSetLayout(dev, descSetLayout, nullptr); descSetLayout = VK_NULL_HANDLE; }

    for (auto& A : atlases) {
        if (A.sampler) vkDestroySampler(dev, A.sampler, nullptr);
        if (A.view)    vkDestroyImageView(dev, A.view, nullptr);
        if (A.image)   vkDestroyImage(dev, A.image, nullptr);
        if (A.memory)  vkFreeMemory(dev, A.memory, nullptr);
    }
    destroyGeometry();
}

// ---- public helpers ---------------------------------------------------------

int TextRenderer::idxForPx(int px) const {
    // choose nearest bucket
    int bestIdx = 0;
    int bestDiff = std::abs(px - bucketPx[0]);
    for (int i = 1; i < (int)bucketPx.size(); ++i) {
        int d = std::abs(px - bucketPx[i]);
        if (d < bestDiff) { bestDiff = d; bestIdx = i; }
    }
    return bestIdx;
}

int TextRenderer::getBucketPx(int pixelSize) const {
    return bucketPx[idxForPx(pixelSize)];
}

const TextRenderer::Atlas* TextRenderer::getAtlasForPx(int pixelSize) const {
    int i = idxForPx(pixelSize);
    if (i < 0 || i >= (int)atlases.size()) return nullptr;
    return &atlases[i];
}

VkDescriptorSet TextRenderer::getDescriptorSetForPx(int pixelSize) const {
    int i = idxForPx(pixelSize);
    return atlases[i].descriptor;
}

std::vector<TextRenderer::GlyphInstance> TextRenderer::layoutASCII(int pixelSize, std::string_view text, glm::vec2 startPx, glm::vec4 colorRGBA, float letterSpacing, float lineHeight) const
{
    const Atlas* A = getAtlasForPx(pixelSize);
    if (!A) return {};

    std::vector<GlyphInstance> out;
    out.reserve(text.size());

    float baseline = startPx.y + A->ascent; // y-down: baseline below top
    float x = startPx.x;

    const float defaultLineAdvance = (A->ascent - A->descent + A->lineGap);
    const float lineAdv = (lineHeight > 0.0f) ? lineHeight : defaultLineAdvance;

    for (unsigned char c : text) {
        if (c == '\n') { baseline += lineAdv; x = startPx.x; continue; }
        if (c < 32 || c >= 127) continue;

        const Glyph& g = A->glyphs[c];

        GlyphInstance inst{};
        // packed offsets place TOP-LEFT of the glyph bitmap relative to the baseline
        float gx = std::round(x + g.offX);
        float gy = std::floor(baseline + g.offY + 0.5f);

        inst.pos   = { gx, gy };         // top-left in px
        inst.size  = { g.w, g.h };       // px size
        inst.uvMin = { g.uvMinX, g.uvMinY };
        inst.uvMax = { g.uvMaxX, g.uvMaxY };
        inst.color = colorRGBA;

        out.push_back(inst);
        x += g.advance + letterSpacing;  // packed advance
    }
    return out;
}

// --- geometry lifetime -------------------------------------------------------

void TextRenderer::initGeometry(uint32_t maxGlyphs) {
    if (quadVB == VK_NULL_HANDLE) createQuadBuffer();
    if (instVB == VK_NULL_HANDLE) createInstanceBuffer(maxGlyphs);
}

void TextRenderer::destroyGeometry() {
    VkDevice dev = device.device();
    if (instMapped) { vkUnmapMemory(dev, instMem); instMapped = nullptr; }
    if (instVB)     { vkDestroyBuffer(dev, instVB, nullptr); instVB = VK_NULL_HANDLE; }
    if (instMem)    { vkFreeMemory(dev, instMem, nullptr);   instMem = VK_NULL_HANDLE; }
    if (quadVB)     { vkDestroyBuffer(dev, quadVB, nullptr); quadVB = VK_NULL_HANDLE; }
    if (quadMem)    { vkFreeMemory(dev, quadMem, nullptr);   quadMem = VK_NULL_HANDLE; }
}

void TextRenderer::createQuadBuffer() {
    struct V { float x,y; };
    static const V kQuadStrip[4] = {
        {-0.5f,-0.5f}, { 0.5f,-0.5f}, {-0.5f, 0.5f}, { 0.5f, 0.5f}
    };
    VkDeviceSize sz = sizeof(kQuadStrip);
    device.createBuffer(sz,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        quadVB, quadMem);
    void* map = nullptr;
    vkMapMemory(device.device(), quadMem, 0, sz, 0, &map);
    std::memcpy(map, kQuadStrip, (size_t)sz);
    vkUnmapMemory(device.device(), quadMem);
}

void TextRenderer::createInstanceBuffer(uint32_t capacity) {
    instCapacity = capacity;
    VkDeviceSize sz = sizeof(GlyphInstance) * capacity;
    device.createBuffer(sz,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        instVB, instMem);
    vkMapMemory(device.device(), instMem, 0, sz, 0, &instMapped);
    currentInstanceCount = 0;
}

// --- per-frame upload/bind/draw ---------------------------------------------

void TextRenderer::updateInstances(const std::vector<GlyphInstance>& glyphs) {
    currentInstanceCount = std::min<uint32_t>((uint32_t)glyphs.size(), instCapacity);
    if (currentInstanceCount == 0) return;
    std::memcpy(instMapped, glyphs.data(), currentInstanceCount * sizeof(GlyphInstance));
}

void TextRenderer::bind(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, int pixelSizeBucket) {
    // Bind both vertex buffers (binding 0 = quad, binding 1 = instances)
    VkBuffer bufs[]     = { quadVB, instVB };
    VkDeviceSize offs[] = { 0,      0      };
    vkCmdBindVertexBuffers(cmd, 0, 2, bufs, offs);

    // Bind the atlas descriptor for this bucket at set=0
    VkDescriptorSet ds = getDescriptorSetForPx(pixelSizeBucket);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, /*firstSet*/0, /*count*/1,
                            &ds, 0, nullptr);
}

void TextRenderer::draw(VkCommandBuffer cmd) const {
    if (currentInstanceCount == 0) return;
    vkCmdDraw(cmd, /*vertexCount*/4, /*instanceCount*/currentInstanceCount, /*firstVertex*/0, /*firstInstance*/0);
}

// ---- CPU atlas build --------------------------------------------------------

static std::vector<uint8_t> readFileBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("TextRenderer: failed to open TTF: " + path);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return data;
}

void TextRenderer::buildAtlasesCPU(const std::string& ttfPath) {
    auto ttf = readFileBin(ttfPath);

    // Init font once (used for metrics)
    stbtt_fontinfo fi;
    if (!stbtt_InitFont(&fi, ttf.data(), stbtt_GetFontOffsetForIndex(ttf.data(), 0)))
        throw std::runtime_error("TextRenderer: stbtt_InitFont failed");

    atlases.clear();
    atlases.resize(bucketPx.size());

    for (size_t i = 0; i < bucketPx.size(); ++i) {
        const int px = bucketPx[i];

        // Heuristic atlas size: for 128px we use 2048x2048, others 1024x1024
        int texW = (px >= 128) ? 2048 : 1024;
        int texH = (px >= 128) ? 2048 : 1024;

        Atlas A{};
        A.pixelHeight = px;
        A.texW = texW;
        A.texH = texH;
        A.pixels.assign(texW * texH, 0);

        stbtt_pack_context pc;
        if (!stbtt_PackBegin(&pc, A.pixels.data(), texW, texH, texW /*stride*/, 4 /*padding*/, nullptr))
            throw std::runtime_error("TextRenderer: stbtt_PackBegin failed");
        stbtt_PackSetOversampling(&pc, 1, 1);

        // Pack ASCII 32..126
        std::vector<stbtt_packedchar> packed(95);
        if (!stbtt_PackFontRange(&pc, ttf.data(), 0, (float)px, 32, 95, packed.data())) {
            stbtt_PackEnd(&pc);
            throw std::runtime_error("PackFontRange failed");
        }
        stbtt_PackEnd(&pc);

        // font v-metrics if you still want baseline for line layout
        int ia,id,il;
        stbtt_GetFontVMetrics(&fi, &ia, &id, &il);
        float scale = stbtt_ScaleForPixelHeight(&fi, (float)px);
        A.ascent  = ia * scale;
        A.descent = id * scale;
        A.lineGap = il * scale;

        for (int c = 32; c < 127; ++c) {
            const auto& p = packed[c - 32];
            Glyph g{};
        
            g.w = float(p.x1 - p.x0);
            g.h = float(p.y1 - p.y0);
        
            const float halfTexelX = 0.5f / float(texW);
            const float halfTexelY = 0.5f / float(texH);
        
            g.uvMinX = float(p.x0) / float(texW) + halfTexelX;
            g.uvMinY = float(p.y0) / float(texH) + halfTexelY;
            g.uvMaxX = float(p.x1) / float(texW) - halfTexelX;
            g.uvMaxY = float(p.y1) / float(texH) - halfTexelY;
        
            g.offX    = p.xoff;       // use packed offsets
            g.offY    = p.yoff;       // (y-down)
            g.advance = p.xadvance;   // use packed advance
        
            A.glyphs[c] = g;
        }

        atlases[i] = std::move(A);
    }
}

// ---- Vulkan upload & helpers ------------------------------------------------

void TextRenderer::drawRange(VkCommandBuffer cmd, uint32_t count, uint32_t firstInstance) const {
    if (count == 0) return;
    vkCmdDraw(cmd, /*vertexCount*/4, /*instanceCount*/count, /*firstVertex*/0, /*firstInstance*/firstInstance);
}

void TextRenderer::ensureInstanceCapacity(uint32_t needed) {
    if (instVB == VK_NULL_HANDLE) {                // first time
        if (quadVB == VK_NULL_HANDLE) createQuadBuffer();
        createInstanceBuffer(needed);
        return;
    }
    if (needed <= instCapacity) return;            // enough already

    // re-create bigger buffer (grow to exactly needed; you can round up if you like)
    VkDevice dev = device.device();
    if (instMapped) { vkUnmapMemory(dev, instMem); instMapped = nullptr; }
    if (instVB)     { vkDestroyBuffer(dev, instVB, nullptr); instVB = VK_NULL_HANDLE; }
    if (instMem)    { vkFreeMemory(dev, instMem, nullptr);   instMem = VK_NULL_HANDLE; }

    createInstanceBuffer(needed);
}

void TextRenderer::createDescriptorPoolAndLayout() {
    VkDevice dev = device.device();

    // Layout: set=0, binding=0, combined image sampler
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    linfo.bindingCount = 1;
    linfo.pBindings = &b;
    if (vkCreateDescriptorSetLayout(dev, &linfo, nullptr, &descSetLayout) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: create desc set layout failed");

    // Pool: one sampler per atlas
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = (uint32_t)atlases.size();

    VkDescriptorPoolCreateInfo pinfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pinfo.maxSets = (uint32_t)atlases.size();
    pinfo.poolSizeCount = 1;
    pinfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(dev, &pinfo, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: create desc pool failed");
}

void TextRenderer::uploadAtlas(Atlas& A) {
    VkDevice dev = device.device();

    // --- No mipmaps for text atlases (simplest + avoids bleed) ---
    A.mipLevels = 1;

    // Create image (R8, sampled + copy dst)
    createImage2D(
        A.texW, A.texH, A.mipLevels,
        VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // no TRANSFER_SRC
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        A.image, A.memory
    );

    // Staging buffer
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    const VkDeviceSize size = static_cast<VkDeviceSize>(A.texW) * static_cast<VkDeviceSize>(A.texH);
    device.createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        staging, stagingMem
    );

    void* map = nullptr;
    vkMapMemory(dev, stagingMem, 0, size, 0, &map);
    std::memcpy(map, A.pixels.data(), static_cast<size_t>(size));
    vkUnmapMemory(dev, stagingMem);

    // Upload: UNDEFINED -> TRANSFER_DST, copy, then -> SHADER_READ_ONLY
    transitionImageLayout(
        A.image, VK_FORMAT_R8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        A.mipLevels
    );

    copyBufferToImage(staging, A.image, A.texW, A.texH);

    transitionImageLayout(
        A.image, VK_FORMAT_R8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        A.mipLevels
    );

    // Cleanup staging
    vkDestroyBuffer(dev, staging, nullptr);
    vkFreeMemory(dev, stagingMem, nullptr);

    // View (levelCount = 1)
    createImageView2D(
        A.image, VK_FORMAT_R8_UNORM,
        A.mipLevels, VK_IMAGE_ASPECT_COLOR_BIT, A.view
    );

    // Sampler: no mips, clamp, linear (or set minFilter = NEAREST if you prefer extra crisp)
    VkSamplerCreateInfo sinfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sinfo.magFilter  = VK_FILTER_LINEAR;
    sinfo.minFilter  = VK_FILTER_LINEAR;                 // try VK_FILTER_NEAREST for pixel-crisp
    sinfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;   // no mip sampling
    sinfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sinfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sinfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sinfo.anisotropyEnable = VK_FALSE;
    sinfo.maxAnisotropy    = 1.0f;
    sinfo.mipLodBias = 0.0f;
    sinfo.minLod     = 0.0f;
    sinfo.maxLod     = 0.0f;                               // lock to base mip
    if (vkCreateSampler(dev, &sinfo, nullptr, &A.sampler) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: create sampler failed");

    // Descriptor (set=0, binding=0)
    VkDescriptorSetAllocateInfo ainfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ainfo.descriptorPool     = descPool;
    ainfo.descriptorSetCount = 1;
    ainfo.pSetLayouts        = &descSetLayout;

    if (vkAllocateDescriptorSets(dev, &ainfo, &A.descriptor) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: alloc descriptor failed");

    VkDescriptorImageInfo di{};
    di.sampler     = A.sampler;
    di.imageView   = A.view;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = A.descriptor;
    w.dstBinding      = 0;
    w.dstArrayElement = 0;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.descriptorCount = 1;
    w.pImageInfo      = &di;

    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

    // Free CPU pixels
    A.pixels.clear();
    A.pixels.shrink_to_fit();
}


// ---- low-level Vulkan helpers ----------------------------------------------

uint32_t TextRenderer::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(device.getPhysicalDevice(), &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ( (typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props )
            return i;
    }
    throw std::runtime_error("TextRenderer: no suitable memory type");
}

void TextRenderer::createImage2D(uint32_t w, uint32_t h, uint32_t mipLevels,
                                 VkFormat format, VkImageTiling tiling,
                                 VkImageUsageFlags usage, VkMemoryPropertyFlags props,
                                 VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.extent = { w, h, 1 };
    info.mipLevels = mipLevels;
    info.arrayLayers = 1;
    info.format = format;
    info.tiling = tiling;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    info.usage = usage;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device.device(), &info, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: createImage failed");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device.device(), image, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);

    if (vkAllocateMemory(device.device(), &ai, nullptr, &memory) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: alloc image memory failed");

    vkBindImageMemory(device.device(), image, memory, 0);
}

void TextRenderer::createImageView2D(VkImage image, VkFormat format, uint32_t mipLevels,
                                     VkImageAspectFlags aspect, VkImageView& view) {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info.format = format;
    info.subresourceRange.aspectMask = aspect;
    info.subresourceRange.baseMipLevel = 0;
    info.subresourceRange.levelCount = mipLevels;
    info.subresourceRange.baseArrayLayer = 0;
    info.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device.device(), &info, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("TextRenderer: createImageView failed");
}

VkCommandBuffer TextRenderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = cmdPool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device.device(), &a, &cmd);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void TextRenderer::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device.device(), cmdPool, 1, &cmd);
}

void TextRenderer::transitionImageLayout(VkImage image, VkFormat format,
                                         VkImageLayout oldLayout, VkImageLayout newLayout,
                                         uint32_t mipLevels) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        // For mipmap generation we handle layout transitions separately
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(cmd);
}

void TextRenderer::copyBufferToImage(VkBuffer src, VkImage image, uint32_t w, uint32_t h) {
    VkCommandBuffer cmd = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0,0,0};
    region.imageExtent = {w,h,1};

    vkCmdCopyBufferToImage(cmd, src, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(cmd);
}

void TextRenderer::generateMipmaps(VkImage image, int32_t texW, int32_t texH, uint32_t mipLevels, VkFormat format) {
    // Check blit support
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(), format, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        // Fallback: transition to SHADER_READ_ONLY without mips
        transitionImageLayout(image, format,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              1);
        return;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();

    int32_t w = texW;
    int32_t h = texH;

    for (uint32_t i = 1; i < mipLevels; ++i) {
        // Transition level i-1: TRANSFER_DST -> TRANSFER_SRC
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Blit from i-1 -> i
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[0] = {0,0,0};
        blit.srcOffsets[1] = {w, h, 1};

        int32_t nw = std::max(1, w / 2);
        int32_t nh = std::max(1, h / 2);

        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0,0,0};
        blit.dstOffsets[1] = {nw, nh, 1};

        vkCmdBlitImage(cmd,
            image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        // Transition i-1 to SHADER_READ_ONLY
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        w = nw; h = nh;
    }

    // Transition last level to SHADER_READ_ONLY
    VkImageMemoryBarrier last{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    last.image = image;
    last.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    last.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    last.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    last.subresourceRange.baseArrayLayer = 0;
    last.subresourceRange.layerCount = 1;
    last.subresourceRange.baseMipLevel = mipLevels - 1;
    last.subresourceRange.levelCount = 1;
    last.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    last.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    last.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    last.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &last);

    endSingleTimeCommands(cmd);
}

std::vector<VkVertexInputBindingDescription> TextRenderer::GlyphInstance::getBindDescriptions() {
    std::vector<VkVertexInputBindingDescription> b(1);
    b[0].binding   = 1;
    b[0].stride    = sizeof(GlyphInstance);
    b[0].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    return b;
}

std::vector<VkVertexInputAttributeDescription> TextRenderer::GlyphInstance::getAttributeDescriptions() {
    std::vector<VkVertexInputAttributeDescription> a(5);
    a[0].binding = 1; 
    a[0].location = 1; 
    a[0].format = VK_FORMAT_R32G32_SFLOAT; 
    a[0].offset = offsetof(GlyphInstance, pos);
    // loc2: size
    a[1].binding = 1; 
    a[1].location = 2; 
    a[1].format = VK_FORMAT_R32G32_SFLOAT; 
    a[1].offset = offsetof(GlyphInstance, size);
    // loc3: uvMin
    a[2].binding = 1; 
    a[2].location = 3; 
    a[2].format = VK_FORMAT_R32G32_SFLOAT; 
    a[2].offset = offsetof(GlyphInstance, uvMin);
    // loc4: uvMax
    a[3].binding = 1; 
    a[3].location = 4; 
    a[3].format = VK_FORMAT_R32G32_SFLOAT; 
    a[3].offset = offsetof(GlyphInstance, uvMax);
    // loc5: color
    a[4].binding = 1; 
    a[4].location = 5; 
    a[4].format = VK_FORMAT_R32G32B32A32_SFLOAT; 
    a[4].offset = offsetof(GlyphInstance, color);
    return a;
}