#pragma once
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#include <algorithm>

class Device;

class TextRenderer {
public:
    // Per-glyph instance (binding 1 in your text pipeline)
    struct GlyphInstance {
      glm::vec2 pos;
      glm::vec2 size;
      glm::vec2 uvMin;
      glm::vec2 uvMax;
      glm::vec4 color;

      static std::vector<VkVertexInputBindingDescription> getBindDescriptions();
      static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
    };


    // Simple glyph metrics & atlas UVs (ASCII)
    struct Glyph {
        float w, h;                   // bitmap size (px)
        float uvMinX, uvMinY;
        float uvMaxX, uvMaxY;

        float offX;                   // from stbtt_packedchar.xoff
        float offY;                   // from stbtt_packedchar.yoff  (y-down)
        float advance;                // from stbtt_packedchar.xadvance
    };

    struct Atlas {
        int                pixelHeight = 0; // baked size
        int                texW = 0, texH = 0;
        float              ascent = 0, descent = 0, lineGap = 0;
        std::array<Glyph, 128> glyphs{}; // ASCII

        // Vulkan resources
        VkImage            image = VK_NULL_HANDLE;
        VkDeviceMemory     memory = VK_NULL_HANDLE;
        VkImageView        view = VK_NULL_HANDLE;
        VkSampler          sampler = VK_NULL_HANDLE;
        uint32_t           mipLevels = 1;
        VkDescriptorSet    descriptor = VK_NULL_HANDLE;

        // CPU pixels only during build
        std::vector<uint8_t> pixels; // R8
    };

public:
    // You pass: device wrapper, a command pool (graphics, resettable) and the graphics queue, plus a TTF file path
    TextRenderer(Device& device, VkCommandPool cmdPool, VkQueue graphicsQueue,
                 const std::string& ttfPath);
    ~TextRenderer();

    TextRenderer(const TextRenderer&) = delete;
    TextRenderer& operator=(const TextRenderer&) = delete;

    // Descriptor set layout (set=0, binding=0, combined image sampler)
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descSetLayout; }

    // Get the descriptor set for a text size (exact match or nearest bucket)
    VkDescriptorSet getDescriptorSetForPx(int pixelSize) const;

    // Which baked size will be used for 'pixelSize'? (nearest)
    int getBucketPx(int pixelSize) const;

    // Quick ASCII layout; lineHeight==0 → use font’s default (ascent-descent+lineGap)
    std::vector<GlyphInstance> layoutASCII(int pixelSize, std::string_view text, glm::vec2 startPx, glm::vec4 colorRGBA, float letterSpacing, float lineHeight) const;

    // Access to atlas metrics (in case you need ascent/lineGap)
    const Atlas* getAtlasForPx(int pixelSize) const;

    void initGeometry(uint32_t maxGlyphs);

    // Per-frame updates & draw (mirror UiRenderer)
    void updateInstances(const std::vector<GlyphInstance>& glyphs);
    void bind(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, int pixelSizeBucket);
    void draw(VkCommandBuffer cmd) const;

    void drawRange(VkCommandBuffer cmd, uint32_t count, uint32_t firstInstance) const;
    void ensureInstanceCapacity(uint32_t needed);
    
    uint32_t getInstanceCapacity() const { return instCapacity; }
    uint32_t instanceCount() const { return currentInstanceCount; }

private:
    // ---- CPU atlas build (stb_truetype) ----
    void buildAtlasesCPU(const std::string& ttfPath);

    // ---- Vulkan upload ----
    void createDescriptorPoolAndLayout();
    void uploadAtlas(Atlas& A); // creates image, view, sampler, descriptor

    // Helpers (immediate commands & resource creation)
    VkCommandBuffer beginSingleTimeCommands();
    void            endSingleTimeCommands(VkCommandBuffer cmd);

    void createImage2D(uint32_t w, uint32_t h, uint32_t mipLevels,
                       VkFormat format, VkImageTiling tiling,
                       VkImageUsageFlags usage, VkMemoryPropertyFlags props,
                       VkImage& image, VkDeviceMemory& memory);

    void createImageView2D(VkImage image, VkFormat format, uint32_t mipLevels,
                           VkImageAspectFlags aspect, VkImageView& view);

    void transitionImageLayout(VkImage image, VkFormat format,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               uint32_t mipLevels);

    void copyBufferToImage(VkBuffer src, VkImage image, uint32_t w, uint32_t h);

    void generateMipmaps(VkImage image, int32_t texW, int32_t texH, uint32_t mipLevels, VkFormat format);

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);

    void*          instMapped    = nullptr;
    
    uint32_t       instCapacity  = 0;
    uint32_t       currentInstanceCount = 0;
    
    void createQuadBuffer();
    void createInstanceBuffer(uint32_t capacity);
    void destroyGeometry();
    
private:
    VkBuffer       quadVB        = VK_NULL_HANDLE;
    VkDeviceMemory quadMem       = VK_NULL_HANDLE;

    VkBuffer       instVB        = VK_NULL_HANDLE;
    VkDeviceMemory instMem       = VK_NULL_HANDLE;
    Device&        device;
    VkCommandPool  cmdPool = VK_NULL_HANDLE;
    VkQueue        graphicsQueue = VK_NULL_HANDLE;

    // descriptor
    VkDescriptorSetLayout descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      descPool = VK_NULL_HANDLE;

    // Atlases for sizes: 16,24,32,64,128
    std::vector<int>      bucketPx {16,24,32,64,128};
    std::vector<Atlas>    atlases; // same size as bucketPx

    // Cached map from px -> index into atlases
    int idxForPx(int px) const;
};
