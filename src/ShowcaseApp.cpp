#include "ShowcaseApp.hpp"
#include <iostream>
#include <cstdio>


constexpr uint32_t NUM_PATH_QUEUES = 4;

// ~~~~~ helpers ~~~~~~

static ImageRGBA8 makeDummyPink1x1()
{
    ImageRGBA8 img;
    img.width  = 1;
    img.height = 1;
    img.pixels = { 255, 0, 255, 255 };
    img.filePath = "dummy://pink";
    return img;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
  std::cerr << "Showcase validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

static std::vector<uint32_t> uniqueIndices(std::initializer_list<std::optional<uint32_t>> list)
{
    std::vector<uint32_t> out;
    out.reserve(list.size());
    std::unordered_set<uint32_t> seen;
    for (const auto& o : list)
    {
        if (!o.has_value())
            continue;
        uint32_t idx = *o;
        if (seen.insert(idx).second)
            out.push_back(idx);
    }
    return out;
}

static void destroyGpuTexture(VkDevice device, GpuTexture& tex) {
    if (tex.view)   { vkDestroyImageView(device, tex.view, nullptr);  tex.view = VK_NULL_HANDLE; }
    if (tex.image)  { vkDestroyImage(device, tex.image, nullptr);     tex.image = VK_NULL_HANDLE; }
    if (tex.memory) { vkFreeMemory(device, tex.memory, nullptr);      tex.memory = VK_NULL_HANDLE; }
}

static void destroyNRDTexture(VkDevice device, NrdTexture& tex) {
    if (tex.view)   { vkDestroyImageView(device, tex.view, nullptr);  tex.view = VK_NULL_HANDLE; }
    if (tex.image)  { vkDestroyImage(device, tex.image, nullptr);     tex.image = VK_NULL_HANDLE; }
    if (tex.memory) { vkFreeMemory(device, tex.memory, nullptr);      tex.memory = VK_NULL_HANDLE; }
}

VkFormat ShowcaseApp::mapNrdFormatToVk(nrd::Format fmt)
{
    switch (fmt)
    {
    case nrd::Format::R8_UNORM:        return VK_FORMAT_R8_UNORM;
    case nrd::Format::R8_SNORM:        return VK_FORMAT_R8_SNORM;
    case nrd::Format::R8_UINT:         return VK_FORMAT_R8_UINT;
    case nrd::Format::R8_SINT:         return VK_FORMAT_R8_SINT;

    case nrd::Format::RG8_UNORM:       return VK_FORMAT_R8G8_UNORM;
    case nrd::Format::RG8_SNORM:       return VK_FORMAT_R8G8_SNORM;
    case nrd::Format::RG8_UINT:        return VK_FORMAT_R8G8_UINT;
    case nrd::Format::RG8_SINT:        return VK_FORMAT_R8G8_SINT;

    case nrd::Format::RGBA8_UNORM:     return VK_FORMAT_R8G8B8A8_UNORM;
    case nrd::Format::RGBA8_SNORM:     return VK_FORMAT_R8G8B8A8_SNORM;
    case nrd::Format::RGBA8_UINT:      return VK_FORMAT_R8G8B8A8_UINT;
    case nrd::Format::RGBA8_SINT:      return VK_FORMAT_R8G8B8A8_SINT;
    case nrd::Format::RGBA8_SRGB:      return VK_FORMAT_R8G8B8A8_SRGB;

    case nrd::Format::R16_UNORM:       return VK_FORMAT_R16_UNORM;
    case nrd::Format::R16_SNORM:       return VK_FORMAT_R16_SNORM;
    case nrd::Format::R16_UINT:        return VK_FORMAT_R16_UINT;
    case nrd::Format::R16_SINT:        return VK_FORMAT_R16_SINT;
    case nrd::Format::R16_SFLOAT:      return VK_FORMAT_R16_SFLOAT;

    case nrd::Format::RG16_UNORM:      return VK_FORMAT_R16G16_UNORM;
    case nrd::Format::RG16_SNORM:      return VK_FORMAT_R16G16_SNORM;
    case nrd::Format::RG16_UINT:       return VK_FORMAT_R16G16_UINT;
    case nrd::Format::RG16_SINT:       return VK_FORMAT_R16G16_SINT;
    case nrd::Format::RG16_SFLOAT:     return VK_FORMAT_R16G16_SFLOAT;

    case nrd::Format::RGBA16_UNORM:    return VK_FORMAT_R16G16B16A16_UNORM;
    case nrd::Format::RGBA16_SNORM:    return VK_FORMAT_R16G16B16A16_SNORM;
    case nrd::Format::RGBA16_UINT:     return VK_FORMAT_R16G16B16A16_UINT;
    case nrd::Format::RGBA16_SINT:     return VK_FORMAT_R16G16B16A16_SINT;
    case nrd::Format::RGBA16_SFLOAT:   return VK_FORMAT_R16G16B16A16_SFLOAT;

    case nrd::Format::R32_UINT:        return VK_FORMAT_R32_UINT;
    case nrd::Format::R32_SINT:        return VK_FORMAT_R32_SINT;
    case nrd::Format::R32_SFLOAT:      return VK_FORMAT_R32_SFLOAT;

    case nrd::Format::RG32_UINT:       return VK_FORMAT_R32G32_UINT;
    case nrd::Format::RG32_SINT:       return VK_FORMAT_R32G32_SINT;
    case nrd::Format::RG32_SFLOAT:     return VK_FORMAT_R32G32_SFLOAT;

    case nrd::Format::RGB32_UINT:      return VK_FORMAT_R32G32B32_UINT;
    case nrd::Format::RGB32_SINT:      return VK_FORMAT_R32G32B32_SINT;
    case nrd::Format::RGB32_SFLOAT:    return VK_FORMAT_R32G32B32_SFLOAT;

    case nrd::Format::RGBA32_UINT:     return VK_FORMAT_R32G32B32A32_UINT;
    case nrd::Format::RGBA32_SINT:     return VK_FORMAT_R32G32B32A32_SINT;
    case nrd::Format::RGBA32_SFLOAT:   return VK_FORMAT_R32G32B32A32_SFLOAT;

    case nrd::Format::R10_G10_B10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case nrd::Format::R10_G10_B10_A2_UINT:  return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case nrd::Format::R11_G11_B10_UFLOAT:   return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case nrd::Format::R9_G9_B9_E5_UFLOAT:   return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;

    default:
        throw std::runtime_error("Showcase: unsupported NRD format");
    }
}


uint32_t ShowcaseApp::getMaxPaths() const
{
    uint32_t w = std::max(1u, rayTraceExtent.width);
    uint32_t h = std::max(1u, rayTraceExtent.height);

    return w * h;
}

VkShaderModule ShowcaseApp::createShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create shader module!");
    return module;
}

uint32_t ShowcaseApp::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props, VkPhysicalDevice phys)
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Showcase: No suitable memory type.");
}

void ShowcaseApp::createBuffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags, VkBuffer& outBuf, VkDeviceMemory& outMem)
{
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size  = size;
    createInfo.usage = usage;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &createInfo, nullptr, &outBuf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, outBuf, &req);

    VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize  = req.size;
    allocateInfo.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memFlags, phys);

    if (vkAllocateMemory(device, &allocateInfo, nullptr, &outMem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");

    vkBindBufferMemory(device, outBuf, outMem, 0);
}

static VkCommandBuffer beginOneTimeCmd(VkDevice device, VkCommandPool pool)
{
    VkCommandBufferAllocateInfo alocateInfo{};
    alocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alocateInfo.commandPool = pool;
    alocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alocateInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alocateInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

static void endOneTimeCmd(VkDevice device, VkQueue queue, VkCommandPool pool, VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

void copyBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
    VkCommandBuffer cmd = beginOneTimeCmd(device, pool);
    VkBufferCopy region{0,0,size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    endOneTimeCmd(device, queue, pool, cmd);
}

void ShowcaseApp::createOrResizeWavefrontBuffers()
{
    if constexpr (SimpleRayTrace)
        return;

    const uint32_t maxPaths = getMaxPaths();
    const VkDeviceSize pathCount = VkDeviceSize(maxPaths);

    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    for (uint32_t frame = 0; frame < SwapChain::MAX_FRAMES_IN_FLIGHT; ++frame)
    {
        ensureBufferCapacity(pathHeaderBuf[frame], pathHeaderMem[frame],
                             pathCount * sizeof(PathHeader),
                             usage, memFlags);

        ensureBufferCapacity(rayBuf[frame], rayMem[frame],
                             pathCount * sizeof(Ray),
                             usage, memFlags);

        ensureBufferCapacity(hitIdsBuf[frame], hitIdsMem[frame],
                             pathCount * sizeof(HitIds),
                             usage, memFlags);

        ensureBufferCapacity(hitDataBuf[frame], hitDataMem[frame],
                             pathCount * sizeof(Hitdata),
                             usage, memFlags);

        ensureBufferCapacity(radianceBuf[frame], radianceMem[frame],
                             pathCount * sizeof(RadianceState),
                             usage, memFlags);

        ensureBufferCapacity(bsdfSampleBuf[frame], bsdfSampleMem[frame],
                             pathCount * sizeof(BsdfSample),
                             usage, memFlags);

        ensureBufferCapacity(lightSampleBuf[frame], lightSampleMem[frame],
                             pathCount * sizeof(LightSample),
                             usage, memFlags);

        ensureBufferCapacity(shadowRayBuf[frame], shadowRayMem[frame],
                             pathCount * sizeof(ShadowRay),
                             usage, memFlags);

        ensureBufferCapacity(shadowResultBuf[frame], shadowResultMem[frame],
                             pathCount * sizeof(ShadowResult),
                             usage, memFlags);

        ensureBufferCapacity(pathQueueBuf[frame], pathQueueMem[frame],
                             VkDeviceSize(NUM_PATH_QUEUES) * sizeof(PathQueue),
                             usage, memFlags);
        
        ensureBufferCapacity(pixelStatsBuf[frame], pixelStatsMem[frame],
                             pathCount * sizeof(PixelStatsGPU),
                             usage, memFlags);

        ensureBufferCapacity(highVarPixelBuf[frame], highVarPixelMem[frame],
                             pathCount * sizeof(uint32_t),
                             usage, memFlags);

        ensureBufferCapacity(adaptiveCountersBuf[frame], adaptiveCountersMem[frame],
                             sizeof(AdaptiveCountersGPU),
                             usage, memFlags);
    }
}


static int getAspectExtent(float aspect, bool isWidth)
{
    if (aspect <= EPSILON)
        return isWidth ? 1200 : 900;

    float width = 1200.0f;
    float height = width / aspect;
    if (height > 1080.0f)
    {
        height = 1080.0f;
        width = aspect * height;
    }
    if (width > 1920.0f)
        return isWidth ? 1920 : static_cast<int>(height);
    return isWidth ? static_cast<int>(width) : static_cast<int>(height);
}

void ShowcaseApp::CharCallback(GLFWwindow*, unsigned int cp)
{
    if (s_active)
        s_active->onChar(cp);
}

void ShowcaseApp::KeyCallback(GLFWwindow*, int key, int, int action, int mods)
{
    if (s_active)
        s_active->onKey(key, action, mods);
}

void ShowcaseApp::onChar(uint32_t cp)
{
    if (cp == 'w')
    {
        scene.camera.position.z -= 0.1;
        scene.camera.target.z -= 0.1;
    }
    if (cp == 's')
    {
        scene.camera.position.z += 0.1;
        scene.camera.target.z += 0.1;
    }
    if (cp == 'a')
    {
        scene.camera.position.x -= 0.1;
        scene.camera.target.x -= 0.1;
    }
    if (cp == 'd')
    {
        scene.camera.position.x += 0.1;
        scene.camera.target.x += 0.1;
    }
    if (cp == 32)
    {
        scene.camera.position.y += 0.1;
        scene.camera.target.y += 0.1;
    }
    
    
}

void ShowcaseApp::onKey(int key, int action, int )
{

    switch (key) {
        case GLFW_KEY_BACKSPACE:
            break;
        case GLFW_KEY_ENTER:
            break;
        case GLFW_KEY_LEFT_SHIFT:
            scene.camera.position.y -= 0.1;
            scene.camera.target.y -= 0.1;
            break;
        case GLFW_KEY_KP_ENTER:
            break;
        case GLFW_KEY_ESCAPE:
            break;
        default:
            break;
    }
}

void ShowcaseApp::MouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS)
        {
            input.rmbDown = true;
            input.firstDrag = true;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else if (action == GLFW_RELEASE)
        {
            input.rmbDown = false;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

void ShowcaseApp::CursorPosCallback(GLFWwindow*, double x, double y)
{
    if (!input.rmbDown)
        return;

    if (input.firstDrag)
    {
        input.firstDrag = false;
        input.lastX = x; input.lastY = y;
        return;
    }

    double dx = x - input.lastX;
    double dy = y - input.lastY;
    input.lastX = x; input.lastY = y;

    input.yawDeg   += float(dx) * input.sensitivity;
    input.pitchDeg -= float(dy) * input.sensitivity;

    input.pitchDeg = std::clamp(input.pitchDeg, -89.9f, 89.9f);

    const float cy = std::cos(glm::radians(input.yawDeg));
    const float sy = std::sin(glm::radians(input.yawDeg));
    const float cp = std::cos(glm::radians(input.pitchDeg));
    const float sp = std::sin(glm::radians(input.pitchDeg));

    glm::vec3 forward = glm::normalize(glm::vec3(
        cp * cy,   // x
        sp,        // y
        cp * sy    // z
    ));

    scene.camera.target = scene.camera.position + forward;
    hasMoved = true;
}


void ShowcaseApp::initLookAnglesFromCamera()
{
    glm::vec3 f = glm::normalize(scene.camera.target - scene.camera.position);
    input.yawDeg   = glm::degrees(std::atan2(f.z, f.x));
    input.pitchDeg = glm::degrees(std::asin(glm::clamp(f.y, -1.f, 1.f)));
}

void ShowcaseApp::initUploadResources()
{
    uint32_t family = hasDedicatedTransfer ? transferFamily : computeFamily;

    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.queueFamilyIndex = family;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &pci, nullptr, &frameUpload[i].uploadPool) != VK_SUCCESS)
            throw std::runtime_error("Showcase App: failed to create command pool");

        VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = frameUpload[i].uploadPool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cai, &frameUpload[i].uploadCB) != VK_SUCCESS)
            throw std::runtime_error("Showcase App: failed to allocate command Buffers");

        VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (vkCreateSemaphore(device, &sci, nullptr, &frameUpload[i].uploadDone) != VK_SUCCESS)
            throw std::runtime_error("Showcase App: failed to create upload resource semaphore");

        VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;                   // NEW
        if (vkCreateFence(device, &fci, nullptr, &frameUpload[i].uploadFence) != VK_SUCCESS)
            throw std::runtime_error("Showcase App: failed to make Frame Upload Fence");
    }
}

void ShowcaseApp::ensureUploadStagingCapacity(uint32_t frameIndex, VkDeviceSize need)
{
    auto& frame = frameUpload[frameIndex];
    if (frame.capacity >= need)
        return;

    if (frame.staging)
    {
        vkDestroyBuffer(device, frame.staging, nullptr);
        frame.staging = VK_NULL_HANDLE;
    }
    if (frame.stagingMem)
    {
        vkFreeMemory(device, frame.stagingMem, nullptr);
        frame.stagingMem = VK_NULL_HANDLE;
    }
    frame.mapped = nullptr;

    VkDeviceSize newCap = std::max(need, frame.capacity == 0 ? VkDeviceSize(256 * 1024) : frame.capacity * 2);

    createBuffer(device, physicalDevice, newCap,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 frame.staging, frame.stagingMem);

    vkMapMemory(device, frame.stagingMem, 0, newCap, 0, &frame.mapped);
    frame.capacity = newCap;
}

void ShowcaseApp::FlattenSceneTexturesAndRemapMaterials()
{
    textureIndexMap.clear();
    std::vector<ImageRGBA8> flat;


    for (auto& object : scene.objects)
    {
        uint32_t m = object.meshID;
        if (m >= scene.meshes.size())
            throw std::runtime_error("ShowcaseApp: object.meshID out of range");
        auto& mesh = scene.meshes[m];

        for (size_t i = 0; i < mesh.textures.size(); i++)
        {
            const ImageRGBA8& img = mesh.textures[i];

            uint32_t globalIdx = 0xFFFFFFFFu;

            if (!img.filePath.empty())
            {
                auto it = textureIndexMap.find(img.filePath);
                if (it != textureIndexMap.end())
                    globalIdx = it->second;
                else
                {
                    globalIdx = static_cast<uint32_t>(flat.size());
                    flat.push_back(img);
                    textureIndexMap.emplace(img.filePath, globalIdx);
                }
            }
            else
                throw std::runtime_error("ShowcaseApp: Unexpected Texture ImageRGBA8 without filepath found");
        }
    }

    for (auto& object : scene.objects)
    {
        uint32_t m = object.meshID;
        if (m >= scene.meshes.size())
            throw std::runtime_error("ShowcaseApp: object.meshID out of range");
        auto& mesh = scene.meshes[m];

        for (auto& material : mesh.perMeshMaterials)
        {
            uint textureIndex = glm::floatBitsToUint(material.textureId.x);
            if (textureIndex == 0xFFFFFFFFu)
                continue;
            if (textureIndex >= mesh.textures.size())
                throw std::runtime_error("ShowcaseApp: material texture index out of range");
            const ImageRGBA8& image = mesh.textures[textureIndex];
            auto it = textureIndexMap.find(image.filePath);
            uint32_t globalIndex;
            if (it != textureIndexMap.end())
                globalIndex = it->second;
            else
                globalIndex = 0xFFFFFFFFu;
            material.textureId.x = glm::uintBitsToFloat(globalIndex);
        }
    }

    flattened = std::move(flat);
}

static void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkAccessFlags srcAccess, VkAccessFlags dstAccess, uint32_t baseMip, uint32_t mipCount)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = oldLayout;
    barrier.newLayout                       = newLayout;
    barrier.srcAccessMask                   = srcAccess;
    barrier.dstAccessMask                   = dstAccess;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.subresourceRange.baseMipLevel   = baseMip;
    barrier.subresourceRange.levelCount     = mipCount;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

static void generateMipChainBlit(VkPhysicalDevice phys, VkCommandBuffer cmd, VkImage image, VkFormat format, int32_t width, int32_t height, uint32_t mipLevels)
{
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(phys, format, &props);
    const bool canBlit = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    if (!canBlit || mipLevels <= 1)
        return;

    int32_t w = width;
    int32_t h = height;

    for (uint32_t level = 1; level < mipLevels; ++level)
    {
        transitionImage(cmd, image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                        level - 1, 1);

        VkImageBlit blit{};
        blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel       = level - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount     = 1;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {std::max(1, w), std::max(1, h), 1};

        const int32_t nw = std::max(1, w >> 1);
        const int32_t nh = std::max(1, h >> 1);

        blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel       = level;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount     = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nw, nh, 1};

        vkCmdBlitImage(cmd,
                       image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        transitionImage(cmd, image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // or FRAGMENT
                        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                        level - 1, 1);

        w = nw;
        h = nh;
    }

    transitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // or FRAGMENT
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    mipLevels - 1, 1);
}

GpuTexture ShowcaseApp::createTextureFromImageRGBA8(const ImageRGBA8& img, VkDevice device, VkPhysicalDevice phys, VkFormat format, bool generateMips)
{
    GpuTexture out{};
    out.width  = static_cast<uint32_t>(img.width);
    out.height = static_cast<uint32_t>(img.height);
    out.format = format;
    out.mipLevels = generateMips ? (1u + uint32_t(std::floor(std::log2(std::max(out.width, out.height))))) : 1u;

    VkImageCreateInfo createInfo{};
    createInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    createInfo.imageType     = VK_IMAGE_TYPE_2D;
    createInfo.format        = format;
    createInfo.extent        = { out.width, out.height, 1 };
    createInfo.mipLevels     = out.mipLevels;
    createInfo.arrayLayers   = 1;
    createInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    createInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    createInfo.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        (generateMips ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &createInfo, nullptr, &out.image) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImage failed");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, out.image, &req);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, phys);
    if (vkAllocateMemory(device, &mai, nullptr, &out.memory) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (image) failed");

    vkBindImageMemory(device, out.image, out.memory, 0);

    const VkDeviceSize byteSize = VkDeviceSize(out.width) * out.height * 4;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;

    createBuffer(device, phys, byteSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);

    void* mapped = nullptr;
    vkMapMemory(device, stagingMem, 0, byteSize, 0, &mapped);
    std::memcpy(mapped, img.pixels.data(), size_t(byteSize));
    vkUnmapMemory(device, stagingMem);

    VkCommandBuffer cmd = beginOneTimeCmd(device, graphicsCommandPool);

    transitionImage(cmd, out.image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    0, out.mipLevels);

    VkBufferImageCopy imageCopy{};
    imageCopy.bufferOffset = 0;
    imageCopy.bufferRowLength   = 0;
    imageCopy.bufferImageHeight = 0;
    imageCopy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopy.imageSubresource.mipLevel       = 0;
    imageCopy.imageSubresource.baseArrayLayer = 0;
    imageCopy.imageSubresource.layerCount     = 1;
    imageCopy.imageOffset = {0,0,0};
    imageCopy.imageExtent = { out.width, out.height, 1 };

    vkCmdCopyBufferToImage(cmd, staging, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);

    if (generateMips && out.mipLevels > 1)
        generateMipChainBlit(phys, cmd, out.image, format, int32_t(out.width), int32_t(out.height), out.mipLevels);
    else 
        transitionImage(cmd, out.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        0, 1);

    endOneTimeCmd(device, graphicsQueue, graphicsCommandPool, cmd);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image    = out.image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format   = format;
    imageViewCreateInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel   = 0;
    imageViewCreateInfo.subresourceRange.levelCount     = out.mipLevels;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &out.view) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImageView failed");

    return out;
}

// ~~~~ constructor / destructor ~~~~

ShowcaseApp::ShowcaseApp(VkPhysicalDevice gpu, VkInstance inst, Scene scene) : window(getAspectExtent(scene.camera.aspect, true), getAspectExtent(scene.camera.aspect, false), "Scoop"), scene(scene)
{
    instance = inst;
    s_active = this;
    glfwSetMouseButtonCallback(window.handle(), [](GLFWwindow* w, int b, int a, int m){ if (s_active) s_active->MouseButtonCallback(w,b,a,m); });
    glfwSetCursorPosCallback(window.handle(), [](GLFWwindow* w, double x, double y){ if (s_active) s_active->CursorPosCallback(w,x,y); });
    // glfwSetCharCallback(window.handle(), &ShowcaseApp::CharCallback);
    // glfwSetKeyCallback (window.handle(), &ShowcaseApp::KeyCallback);
    setupDebugMessenger();
    window.createWindowSurface(instance, &surface);
    if (gpu == VK_NULL_HANDLE)
        throw std::runtime_error("Showcase: invalid GPU passed");
    physicalDevice = gpu;
    FlattenSceneTexturesAndRemapMaterials();
    createLogicalDevice();
    createQueryPool();
    initUploadResources();
    createTextureSampler(gpu);
    createSwapchain();
    createImageViews();
    createSyncObjects();
    createOffscreenTargets();
    if constexpr (!SimpleRayTrace)
    {
        createFSRTargets();
        createNrdTargets();
        initNRD();
        createNrdInputTargets();
    }
    createParamsBuffers();
    if constexpr (!SimpleRayTrace)
        createFsrConstBuffers();
    createComputeDescriptors();
    createGraphicsDescriptors();
    updateComputeDescriptor();
    createComputePipeline();
    createRenderPass();
    createFramebuffers();
    createFullscreenGraphicsPipeline();
    createWavefrontBuffers();
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(gpu, &properties);
    initLookAnglesFromCamera();
    makeInstances(ShowcaseApp::scene);
    makeEmissionTriangles();
    QueueFamiliyIndies ind = findQueueFamilies(gpu);
    // DumpScene(ShowcaseApp::scene);
}

ShowcaseApp::~ShowcaseApp()
{
    if (graphicsPipeline)
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
    if (graphicsPipelineLayout)
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);

    destroyGraphicsDescriptors();
    destroyFramebuffers();
    
    if (renderPass)
        vkDestroyRenderPass(device, renderPass, nullptr);

    if (computePipeline)
        vkDestroyPipeline(device, computePipeline, nullptr);

    if (rayTraceLogicPipeline)
        vkDestroyPipeline(device, rayTraceLogicPipeline, nullptr);
    if (rayTraceNewPathPipeline)
        vkDestroyPipeline(device, rayTraceNewPathPipeline, nullptr);
    if (rayTraceMaterialPipeline)
        vkDestroyPipeline(device, rayTraceMaterialPipeline, nullptr);
    if (rayTraceExtendRayPipeline)
        vkDestroyPipeline(device, rayTraceExtendRayPipeline, nullptr);
    if (rayTraceShadowRayPipeline)
        vkDestroyPipeline(device, rayTraceShadowRayPipeline, nullptr);
    if (rayTraceFinalWritePipeline)
        vkDestroyPipeline(device, rayTraceFinalWritePipeline, nullptr);
    if (FSRPipeline)
        vkDestroyPipeline(device, FSRPipeline, nullptr);
    if (FSRSharpenPipeline)
        vkDestroyPipeline(device, FSRSharpenPipeline, nullptr);
    if (computePipelineLayout)
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
    
    destroyComputeDescriptors();
    destroyOffscreenTarget();
    if constexpr (!SimpleRayTrace)
    {
        destroyFSRTarget();
        destroyNrdTargets();
        destroyNRD();
    }

    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
    destroySceneTextures();
    if (textureSampler)
    {
        vkDestroySampler(device, textureSampler, nullptr);
        textureSampler = VK_NULL_HANDLE;
    }


    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (paramsMapped[i])
        {
            vkUnmapMemory(device, paramsMemory[i]);
            paramsMapped[i] = nullptr;
        }
        if (paramsBuffer[i]) 
            vkDestroyBuffer(device, paramsBuffer[i], nullptr);
        if (paramsMemory[i])
            vkFreeMemory(device, paramsMemory[i], nullptr);
        if (fsrConstMapped[i])
        {
            vkUnmapMemory(device, fsrConstMemory[i]);
            fsrConstMapped[i] = nullptr;
        }
        if (fsrConstBuffer[i])
            vkDestroyBuffer(device, fsrConstBuffer[i], nullptr);
        if (fsrConstMemory[i])
            vkFreeMemory(device, fsrConstMemory[i], nullptr);
        if (tlasNodesBuf[i]) vkDestroyBuffer(device, tlasNodesBuf[i], nullptr);
        if (tlasNodesMem[i]) vkFreeMemory(device, tlasNodesMem[i], nullptr);
        if (tlasInstBuf[i])  vkDestroyBuffer(device, tlasInstBuf[i], nullptr);
        if (tlasInstMem[i])  vkFreeMemory(device, tlasInstMem[i], nullptr);
        if (tlasIdxBuf[i])   vkDestroyBuffer(device, tlasIdxBuf[i], nullptr);
        if (tlasIdxMem[i])   vkFreeMemory(device, tlasIdxMem[i], nullptr);
        auto& f = frameUpload[i];

        if (f.mapped) { vkUnmapMemory(device, f.stagingMem); f.mapped = nullptr; }
        if (f.staging)     { vkDestroyBuffer(device, f.staging, nullptr);      f.staging = VK_NULL_HANDLE; }
        if (f.stagingMem)  { vkFreeMemory(device, f.stagingMem, nullptr);      f.stagingMem = VK_NULL_HANDLE; }

        if (f.uploadCB)    { vkFreeCommandBuffers(device, f.uploadPool, 1, &f.uploadCB); f.uploadCB = VK_NULL_HANDLE; }
        if (f.uploadPool)  { vkDestroyCommandPool(device, f.uploadPool, nullptr);        f.uploadPool = VK_NULL_HANDLE; }

        if (f.uploadDone)  { vkDestroySemaphore(device, f.uploadDone, nullptr); f.uploadDone = VK_NULL_HANDLE; }
        if (f.uploadFence) { vkDestroyFence(device, f.uploadFence, nullptr);    f.uploadFence = VK_NULL_HANDLE; }
        if (computeFences[i]) vkDestroyFence(device, computeFences[i], nullptr);
        if (imageAcquiredFences[i]) vkDestroyFence(device, imageAcquiredFences[i], nullptr);
    }
    destroyWavefrontBuffers();
    destroySSBOdata();

    for (auto s : imageRenderFinished)
        vkDestroySemaphore(device, s, nullptr);
    imageRenderFinished.clear();
    
    for (size_t i = 0; i < imageAvailableSemaphores.size(); ++i)
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i)
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    for (size_t i = 0; i < inFlightFences.size(); ++i)
        vkDestroyFence(device, inFlightFences[i], nullptr);
    
    for (auto imageView : swapChainImageViews)
        vkDestroyImageView(device, imageView, nullptr);
    
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    if (device)
    {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
    if (VALIDATE)
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
}

ShowcaseApp* ShowcaseApp::s_active = nullptr;

// ~~~~~ Creation ~~~~~

void ShowcaseApp::createWavefrontBuffers()
{
    if constexpr (SimpleRayTrace)
        return;

    destroyWavefrontBuffers();

    createOrResizeWavefrontBuffers();

    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorBufferInfo b0 { pathHeaderBuf[i],         0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b1 { rayBuf[i],                0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b2 { hitIdsBuf[i],             0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b3 { hitDataBuf[i],            0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b4 { radianceBuf[i],           0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b5 { bsdfSampleBuf[i],         0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b6 { lightSampleBuf[i],        0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b7 { shadowRayBuf[i],          0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b8 { shadowResultBuf[i],       0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b9 { pathQueueBuf[i],          0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b10{ pixelStatsBuf[i],        0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b11{ highVarPixelBuf[i],      0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo b12{ adaptiveCountersBuf[i],  0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[13]{};

        auto initWrite = [&](VkWriteDescriptorSet& w, uint32_t binding, const VkDescriptorBufferInfo* info)
        {
            w = {};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = computeDynamicSets[i];
            w.dstBinding      = binding;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo     = info;
        };

        initWrite(writes[0], 0, &b0);
        initWrite(writes[1], 1, &b1);
        initWrite(writes[2], 2, &b2);
        initWrite(writes[3], 3, &b3);
        initWrite(writes[4], 4, &b4);
        initWrite(writes[5], 5, &b5);
        initWrite(writes[6], 6, &b6);
        initWrite(writes[7], 7, &b7);
        initWrite(writes[8], 8, &b8);
        initWrite(writes[9], 9, &b9);
        initWrite(writes[10], 10, &b10);
        initWrite(writes[11], 11, &b11);
        initWrite(writes[12], 12, &b12);

        vkUpdateDescriptorSets(device, 13, writes, 0, nullptr);
    }
}

void ShowcaseApp::createTextureSampler(VkPhysicalDevice phys, float requestedAniso)
{
    if (textureSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(device, textureSampler, nullptr);
        textureSampler = VK_NULL_HANDLE;
    }

    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(phys, &feats);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);

    const bool enableAniso = feats.samplerAnisotropy == VK_TRUE && requestedAniso > 1.0f;
    const float maxAniso = enableAniso ? std::min(requestedAniso, props.limits.maxSamplerAnisotropy) : 1.0f;

    VkSamplerCreateInfo createInfo{};
    createInfo.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    createInfo.magFilter    = VK_FILTER_LINEAR;
    createInfo.minFilter    = VK_FILTER_LINEAR;
    createInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    createInfo.anisotropyEnable = enableAniso ? VK_TRUE : VK_FALSE;
    createInfo.maxAnisotropy    = maxAniso;

    createInfo.minLod     = 0.0f;
    createInfo.maxLod     = VK_LOD_CLAMP_NONE;
    createInfo.mipLodBias = 0.0f;

    createInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    createInfo.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(device, &createInfo, nullptr, &textureSampler) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSampler failed");
}


void ShowcaseApp::createQueryPool()
{
    if (queryPool)
    {
        vkDestroyQueryPool(device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
    }

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> fams(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, fams.data());

    auto checkFam = [&](uint32_t famIdx, const char* name) -> uint32_t
    {
        if (famIdx >= familyCount)
            throw std::runtime_error(std::string("Showcase: invalid ") + name + " queue family index");
        uint32_t bits = fams[famIdx].timestampValidBits;
        if (bits == 0) {
            std::cerr << "Showcase warning: " << name
                      << " queue family does not support timestamp queries.\n";
        }
        return bits;
    };

    const uint32_t gfxBits = checkFam(graphicsFamily, "graphics");
    const uint32_t cmpBits = checkFam(computeFamily,  "compute");

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    timestampPeriodNs = props.limits.timestampPeriod;

    constexpr uint32_t kQueriesPerQueue  = 2;
    constexpr uint32_t kQueuesToTime     = 2;
    constexpr uint32_t kQueriesPerFrame  = kQueriesPerQueue * kQueuesToTime;
    const uint32_t     totalQueries      = kQueriesPerFrame * SwapChain::MAX_FRAMES_IN_FLIGHT;

    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    ci.queryCount = totalQueries;

    if (vkCreateQueryPool(device, &ci, nullptr, &queryPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create timestamp query pool");
}


void ShowcaseApp::makeInstances(Scene& scene)
{
    instances.reserve(scene.objects.size());
    size_t size = scene.meshes.size();
    std::vector<uint32_t> nodeBases(size, 0u);
    std::vector<uint32_t> triBases(size, 0u);
    std::vector<uint32_t> shadeTriBases(size, 0u);
    std::vector<uint32_t> shadeTriCounts(size, 0u);
    std::vector<uint32_t> materialBases(size, 0u);
    std::vector<uint32_t> textureBases(size, 0u);

    uint32_t runningNodeSize = 0;
    uint32_t runningTriSize = 0;
    uint32_t runningShaderSize = 0;
    uint32_t runningMatSize = 0;
    uint32_t runningTextureSize = 0;
    int i = 0;
    for (auto& mesh : scene.meshes)
    {
        if (i != 0)
        {
            nodeBases[i] = nodeBases[i - 1] + runningNodeSize;
            triBases[i] = triBases[i - 1] + runningTriSize;
            shadeTriBases[i] = shadeTriBases[i - 1] + runningShaderSize;
            materialBases[i] = materialBases[i - 1] + runningMatSize;
            textureBases[i] = textureBases[i - 1] + runningTextureSize;
        }
        
        runningNodeSize = static_cast<uint32_t>(mesh.bottomLevelAccelerationStructure.nodes.size());
        runningTriSize = static_cast<uint32_t>(mesh.bottomLevelAccelerationStructure.triangles.intersectionTriangles.size());
        runningShaderSize = static_cast<uint32_t>(mesh.bottomLevelAccelerationStructure.triangles.shadingTriangles.size());
        runningMatSize = static_cast<uint32_t>(mesh.perMeshMaterials.size());
        runningTextureSize = static_cast<uint32_t>(mesh.textures.size());
        shadeTriCounts[i] = runningShaderSize;
        i++;
    }
    SBVHNodes.reserve(nodeBases[i - 1] + runningNodeSize);
    intersectionTrinagles.reserve(triBases[i - 1] + runningTriSize);
    shadingTriangles.reserve(shadeTriBases[i - 1] + runningShaderSize);
    materials.reserve(materialBases[i - 1] + runningMatSize);
    // textures? .reserve(textureBases[i - 1] + runningTextureSize);
    for (auto& mesh : scene.meshes) {
        auto& sbvh = mesh.bottomLevelAccelerationStructure;

        SBVHNodes.insert(SBVHNodes.end(),
            std::make_move_iterator(sbvh.nodes.begin()),
            std::make_move_iterator(sbvh.nodes.end()));
        sbvh.nodes.clear();

        intersectionTrinagles.insert(intersectionTrinagles.end(),
            std::make_move_iterator(sbvh.triangles.intersectionTriangles.begin()),
            std::make_move_iterator(sbvh.triangles.intersectionTriangles.end()));
        sbvh.triangles.intersectionTriangles.clear();

        shadingTriangles.insert(shadingTriangles.end(),
            std::make_move_iterator(sbvh.triangles.shadingTriangles.begin()),
            std::make_move_iterator(sbvh.triangles.shadingTriangles.end()));
        sbvh.triangles.shadingTriangles.clear();

        materials.insert(materials.end(),
            std::make_move_iterator(mesh.perMeshMaterials.begin()),
            std::make_move_iterator(mesh.perMeshMaterials.end()));
        mesh.perMeshMaterials.clear();
    }

    for (auto& object : scene.objects)
    {
        InstanceData inst{};
        inst.modelToWorld = object.transform.affineTransform();
        inst.worldToModel = affineInverse(inst.modelToWorld);
        inst.worldAABB = object.boundingBox;

        inst.nodeBase = nodeBases[object.meshID];
        inst.triBase = triBases[object.meshID];
        inst.shadeTriBase = shadeTriBases[object.meshID];
        inst.shadeTriCount = shadeTriCounts[object.meshID];
        inst.materialBase = materialBases[object.meshID];
        inst.textureBase = textureBases[object.meshID];
        instances.push_back(inst);
    }
}

inline bool isEmissive(glm::vec4 emission)
{
    if (emission.x <= 1e-3f && emission.y <= 1e-3f && emission.z <= 1e-3f)
        return false;
    return true;
}


void ShowcaseApp::makeEmissionTriangles()
{
    std::vector<float> power;
    uint32_t globalBase = 0;
    for (int size = 0; size < instances.size(); size++)
    {
        auto& instance = instances[size];
        instance.lightMapBase = globalBase;
        for (uint32_t i = 0; i < instance.shadeTriCount; i++)
        {
            triToLightIdx.push_back(0xFFFFFFFFu);
            uint32_t mapIdx = globalBase + i;
            uint32_t matIndex = glm::floatBitsToUint(shadingTriangles[instance.shadeTriBase + i].texture_materialId.w);
            glm::vec4 emission = materials[instance.materialBase + matIndex].emission_flags;
            if (!isEmissive(emission))
                continue;
            EmisiveTriangle eTri{};
            eTri.primitiveIndex = instance.shadeTriBase + i;
            eTri.instanceIndex = static_cast<uint32_t>(size);
            eTri.materialIndex = instance.materialBase + matIndex;
            eTri.padding = 69;

            uint32_t lightIdx = static_cast<uint32_t>(emissiveTrinagles.size());
            emissiveTrinagles.push_back(eTri);
            triToLightIdx[mapIdx] = lightIdx;

            auto& mollerTri = intersectionTrinagles[instance.triBase + i];
            float area = 0.5 * glm::length(glm::cross(affineTransformDirection(instance.modelToWorld,glm::vec3(mollerTri.edge_vec1)), affineTransformDirection(instance.modelToWorld,glm::vec3(mollerTri.edge_vec2))));
            float lum = 0.2126*emission.r + 0.7152*emission.g + 0.0722*emission.b;
            float w = lum * area;
            power.push_back(std::max(w, 1e-8f)); 
        }
        globalBase += instance.shadeTriCount;
    }
    if (emissiveTrinagles.empty())
    {
        if constexpr (!SimpleRayTrace)
            throw std::runtime_error("ShowCaseApp: No emissive trinagles means you'll see nothing");
        else
        {
            lightPdf.clear();
            lightProb.clear();
            lightAlias.clear();
            triToLightIdx.clear();
            return;
        }
    }
    emissiveTrinagles[0].padding = static_cast<uint>(emissiveTrinagles.size());

    const size_t N = power.size();
    lightPdf.clear();
    lightProb.clear();
    lightAlias.clear();

    if (N == 0)
        return;

    lightPdf.resize(N);
    lightProb.resize(N);
    lightAlias.resize(N);

    float totalPower = 0.0f;
    for (float w : power)
        totalPower += w;

    if (totalPower <= 0.0f)
    {
        float uniformPdf = 1.0f / float(N);
        for (size_t i = 0; i < N; ++i) {
            lightPdf[i]   = uniformPdf;
            lightProb[i]  = 1.0f;
            lightAlias[i] = static_cast<uint32_t>(i);
        }
        return;
    }

    std::vector<float> scaled(N);
    for (size_t i = 0; i < N; i++)
    {
        float p = power[i] / totalPower;
        lightPdf[i] = p;
        scaled[i] = p * float(N);
    }

    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(N);
    large.reserve(N);

    for (uint32_t i = 0; i < N; ++i) {
        if (scaled[i] <= 1.0f)
            small.push_back(i);
        else
            large.push_back(i);
    }

    while (!small.empty() && !large.empty())
    {
        uint32_t j = small.back();
        small.pop_back();
        uint32_t k = large.back();
        large.pop_back();

        lightProb[j]  = scaled[j];
        lightAlias[j] = k;
        scaled[k] = (scaled[k] + scaled[j]) - 1.0f;
        if (scaled[k] <= 1.0f)
            small.push_back(k);
        else
            large.push_back(k);
    }

    for (uint32_t i : large)
    {
        lightProb[i]  = 1.0f;
        lightAlias[i] = i;
    }

    for (uint32_t i : small)
    {
        lightProb[i]  = 1.0f;
        lightAlias[i] = i;
    }
}

void ShowcaseApp::uploadTextureImages()
{
    gpuTextures.reserve(flattened.size());

    VkFormat fmt = VK_FORMAT_R8G8B8A8_SRGB;
    for (auto& image : flattened)
        gpuTextures.push_back(createTextureFromImageRGBA8(image, device, physicalDevice, fmt, true));

    if (gpuTextures.empty())
        gpuTextures.push_back(createTextureFromImageRGBA8(makeDummyPink1x1(), device, physicalDevice, fmt, false));
    
    if (textureSampler == VK_NULL_HANDLE)
        throw std::runtime_error("ShowcaseApp: textureSampler is null");

    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(gpuTextures.size());
    for (const auto& texture : gpuTextures)
    {
        if (texture.view == VK_NULL_HANDLE)
            throw std::runtime_error("ShowcaseApp: texture has null VkImageView");
        VkDescriptorImageInfo di{};
        di.sampler     = textureSampler;
        di.imageView   = texture.view;
        di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        infos.push_back(di);
    }

    if (computeStaticSet != VK_NULL_HANDLE)
    {
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = computeStaticSet;
        w.dstBinding      = 9;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = static_cast<uint32_t>(infos.size());
        w.pImageInfo      = infos.data();

        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
}

void ShowcaseApp::createLogicalDevice()
{
    QueueFamiliyIndies indices = findQueueFamilies(physicalDevice);
    if (!indices.isComplete())
        throw std::runtime_error("Showcase: Queue families incomplete.");

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> props(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, props.data());

    const uint32_t gfxFam = indices.graphicsFamily.value();
    const uint32_t cmpFam = indices.computeFamily.value();
    const uint32_t prsFam = indices.presentFamily.value();

    bool presentSharesGraphics = (prsFam == gfxFam);

    uint32_t xferFam = cmpFam;
    bool transferIsDistinct = false;
    bool canSplitComputeXfer = false;

    if (indices.transferFamily.has_value())
    {
        xferFam = indices.transferFamily.value();
        if (indices.hasDedicatedTransfer && xferFam != gfxFam && xferFam != cmpFam && xferFam != prsFam)
            transferIsDistinct = true;
        else if (xferFam != gfxFam && xferFam != cmpFam && xferFam != prsFam)
            transferIsDistinct = true;
        else if (xferFam == cmpFam && props[cmpFam].queueCount >= 2)
            canSplitComputeXfer = true;
        else
            xferFam = cmpFam;
    }

    std::unordered_map<uint32_t, uint32_t> requested;
    auto req_at_least = [&](uint32_t fam, uint32_t n)
    {
        uint32_t& cur = requested[fam];
        if (cur < n)
            cur = n;
    };

    req_at_least(gfxFam, 1);
    if (!presentSharesGraphics)
        req_at_least(prsFam, 1);
    req_at_least(cmpFam, 1);

    if (transferIsDistinct)
        req_at_least(xferFam, 1);
    else if (canSplitComputeXfer)
        req_at_least(cmpFam, 2);
    else
        xferFam = cmpFam;

    // Clamp to available queues
    for (auto& kv : requested)
    {
        uint32_t fam  = kv.first;
        uint32_t want = kv.second;
        uint32_t have = props[fam].queueCount;
        if (want > have)
        {
            kv.second = have;
            if (fam == cmpFam && want >= 2 && have < 2)
            {
                canSplitComputeXfer = false;
                xferFam = cmpFam;
            }
        }
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<std::vector<float>> priorities;
    priorities.reserve(requested.size());

    for (auto& kv : requested)
    {
        uint32_t fam = kv.first;
        uint32_t cnt = kv.second;
        if (!cnt) continue;

        priorities.emplace_back(cnt, 1.0f);
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = fam;
        qci.queueCount       = cnt;
        qci.pQueuePriorities = priorities.back().data();
        queueCreateInfos.push_back(qci);
    }

    // ---- Feature chain ----
    // descriptor indexing
    VkPhysicalDeviceDescriptorIndexingFeatures descIdxFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES
    };
    descIdxFeat.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    descIdxFeat.runtimeDescriptorArray                    = VK_TRUE;
    descIdxFeat.descriptorBindingPartiallyBound           = VK_TRUE;

    // buffer device address
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES
    };
    bdaFeat.bufferDeviceAddress = VK_TRUE;

    // synchronization2 (VK_KHR_synchronization2 / Vulkan 1.3 core)
    VkPhysicalDeviceSynchronization2Features sync2Feat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
    };
    sync2Feat.synchronization2 = VK_TRUE;

    // Chain them: deviceFeatures -> sync2 -> descIdx -> bda
    VkPhysicalDeviceFeatures2 deviceFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    deviceFeatures.features.samplerAnisotropy   = VK_TRUE;
    deviceFeatures.features.robustBufferAccess  = VK_TRUE;

    sync2Feat.pNext        = &descIdxFeat;
    descIdxFeat.pNext      = &bdaFeat;
    deviceFeatures.pNext   = &sync2Feat;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos    = queueCreateInfos.data();
    createInfo.pNext                = &deviceFeatures;

    createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (VALIDATE)
    {
        createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Showcase App: failed to make logical device");

    // Grab queues
    vkGetDeviceQueue(device, gfxFam, 0, &graphicsQueue);

    if (presentSharesGraphics)
        presentQueue = graphicsQueue;
    else
        vkGetDeviceQueue(device, prsFam, 0, &presentQueue);

    vkGetDeviceQueue(device, cmpFam, 0, &computeQueue);

    if (transferIsDistinct)
        vkGetDeviceQueue(device, xferFam, 0, &transferQueue);
    else if (canSplitComputeXfer && requested[cmpFam] >= 2)
        vkGetDeviceQueue(device, cmpFam, 1, &transferQueue);
    else
        transferQueue = computeQueue;

    graphicsFamily      = gfxFam;
    presentFamily       = prsFam;
    computeFamily       = cmpFam;
    transferFamily      = xferFam;
    hasDedicatedTransfer = transferIsDistinct && indices.hasDedicatedTransfer;
}



QueueFamiliyIndies ShowcaseApp::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamiliyIndies indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    std::optional<uint32_t> dedicatedTransferCandidate;
    std::optional<uint32_t> anyTransferCandidate;

    int i = 0;
    for (const auto &queueFamily : queueFamilies)
    {
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
            indices.computeFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (queueFamily.queueCount > 0 && presentSupport)
            indices.presentFamily = i;

        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)
        {
            const bool isDedicated = !(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && !(queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT);
            if (isDedicated && !dedicatedTransferCandidate.has_value())
                dedicatedTransferCandidate = i;
            if (!anyTransferCandidate.has_value())
                anyTransferCandidate = i;
        }
        i++;
    }
    if (dedicatedTransferCandidate.has_value())
    {
        indices.transferFamily     = dedicatedTransferCandidate;
        indices.hasDedicatedTransfer = true;
        indices.hasSeparateTransfer  = true;
    }
    else if (anyTransferCandidate.has_value())
    {
        const uint32_t transferFamily = anyTransferCandidate.value();
        if ((!indices.computeFamily.has_value() || transferFamily != indices.computeFamily.value()) &&
            (!indices.graphicsFamily.has_value() || transferFamily != indices.graphicsFamily.value()) &&
            (!indices.presentFamily.has_value()  || transferFamily != indices.presentFamily.value()))
        {
            indices.transferFamily    = transferFamily;
            indices.hasSeparateTransfer = true;
        }
        else
            indices.transferFamily = transferFamily;
    }
    if (indices.transferFamily.has_value() &&
        indices.computeFamily.has_value() &&
        indices.transferFamily.value() == indices.computeFamily.value())
    {
        const auto& queueFamily = queueFamilies[indices.computeFamily.value()];
        if  (queueFamily.queueCount >= 2)
            indices.canSplitComputeXfer = true;
    }
    return indices;
}


void ShowcaseApp::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) 
{
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
  createInfo.pUserData = nullptr;
}

void ShowcaseApp::setupDebugMessenger() 
{
    if (!VALIDATE)
        return;
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("Showcase: failed to set up debug messenger!");
    }
}

SwapChainSupportDetails ShowcaseApp::querrySwapchaindetails(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;
    
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentCount ,nullptr);
    if (presentCount != 0)
    {
        details.presentModes.resize(presentCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentCount ,details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR ShowcaseApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return availableFormat;
    }

    return availableFormats[0];
}

VkPresentModeKHR ShowcaseApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            return availablePresentMode;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D ShowcaseApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;
    else
    {
        VkExtent2D actualExtent = window.getExtent();
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));   
        return actualExtent;
    }
}

void ShowcaseApp::createSwapchain()
{
    SwapChainSupportDetails swapChainSupport = querrySwapchaindetails(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    VkSwapchainKHR oldSwap = validOldSwapchain();

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamiliyIndies indices = findQueueFamilies(physicalDevice);
    std::vector<uint32_t> q = uniqueIndices({indices.graphicsFamily, indices.presentFamily, indices.computeFamily});

    if (q.size() > 1)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = uint32_t(q.size());
        createInfo.pQueueFamilyIndices = q.data();
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwap;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapchain) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create swap chain!");

    if (oldSwap != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(device, oldSwap, nullptr);

    swapChain = newSwapchain;

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
    if constexpr (SimpleRayTrace)
    {
        rayTraceExtent = extent; // simple path traces at display resolution
    }
    else
    {
        rayTraceExtent.width  = std::max(1u, uint32_t(extent.width  * resolutionScale));
        rayTraceExtent.height = std::max(1u, uint32_t(extent.height * resolutionScale));
    }
}

void ShowcaseApp::recreateSwapchain()
{
    vkDeviceWaitIdle(device);

    VkExtent2D extent = window.getExtent();
    while (extent.width == 0 || extent.height == 0) {
        glfwWaitEvents();
        extent = window.getExtent();
    }
    vkDeviceWaitIdle(device);
    destroyFramebuffers();

    for (auto iv : swapChainImageViews)
        vkDestroyImageView(device, iv, nullptr);
    swapChainImageViews.clear();

    for (auto s : imageRenderFinished)
        vkDestroySemaphore(device, s, nullptr);
    imageRenderFinished.clear();

    if (graphicsPipeline)
    {
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (graphicsPipelineLayout)
    {
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (renderPass)
    {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
    destroyOffscreenTarget();
    if constexpr (!SimpleRayTrace)
    {
        destroyFSRTarget();
        destroyNrdTargets();
    }

    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createOffscreenTargets();
    if constexpr (!SimpleRayTrace)
    {
        createFSRTargets();
        createNrdTargets();
        createNrdInputTargets();
    }

    updateComputeDescriptor();
    createWavefrontBuffers();

    if (graphicsDescPool && graphicsSetLayout)
    {
        for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorImageInfo img{};
            img.sampler     = offscreenSampler;
            img.imageView   = (SimpleRayTrace ? offscreenView[i] : fsrView[i]);
            img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = graphicsSets[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo = &img;
    
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }
    }

    createFullscreenGraphicsPipeline();

    imageRenderFinished.resize(swapChainImages.size());
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < imageRenderFinished.size(); ++i)
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageRenderFinished[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create per-image present semaphore!");

    imagesInFlight.assign(swapChainImages.size(), VK_NULL_HANDLE);
    for (int i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
        offscreenInitialized[i] = false;

    window.resetWindowResizedFlag();
}

void ShowcaseApp::createImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.image = swapChainImages[i];
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
           throw std::runtime_error("Showcase: failed to create image views!");
    }

}

void ShowcaseApp::createSyncObjects()
{
    imageAvailableSemaphores.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);

    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    
    imageRenderFinished.resize(swapChainImages.size());

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFenceCreateInfo acqFenceCI{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    acqFenceCI.flags = 0;

    
    for (size_t i = 0; i < imageRenderFinished.size(); ++i)
    {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageRenderFinished[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create per-image present semaphore!");
    }

    for (int i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create imageAvailable semaphore!");

        if (vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create renderFinished semaphore!");

        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create inFlight fence!");
        
        if (vkCreateFence(device, &fenceInfo, nullptr, &computeFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create compute fence!");
        
        if (vkCreateFence(device, &acqFenceCI, nullptr, &imageAcquiredFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase:failed to create image acquired fence");
    }
}

void ShowcaseApp::createOffscreenTargets()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        offscreenValid[i] = false;
        VkExtent3D extent3D{};
        extent3D.width  = rayTraceExtent.width;
        extent3D.height = rayTraceExtent.height;
        extent3D.depth  = 1;

        VkImageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        createInfo.imageType = VK_IMAGE_TYPE_2D;
        createInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        createInfo.extent = extent3D;
        createInfo.mipLevels = 1;
        createInfo.arrayLayers = 1;
        createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        createInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        
        uint32_t fams[2] = { graphicsFamily, computeFamily };
        uint32_t unique[2];
        uint32_t uniqueCount = 0;

        for (uint32_t j = 0; j < 2; ++j)
        {
            uint32_t f = fams[j];
            bool seen = false;
            for (uint32_t k = 0; k < uniqueCount; ++k)
            {
                if (unique[k] == f) { seen = true; break; }
            }
            if (!seen)
                unique[uniqueCount++] = f;
        }

        if (uniqueCount > 1)
        {
            createInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = uniqueCount;
            createInfo.pQueueFamilyIndices   = unique;
        }
        else
        {
            createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices   = nullptr;
        }
        
        
        

        if (vkCreateImage(device, &createInfo, nullptr, &offscreenImage[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create offscreen image!");
        {
            char name[64];
            std::snprintf(name, sizeof(name), "OffscreenImage[%u]", i);
            setImageName(offscreenImage[i], name);
        }
        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(device, offscreenImage[i], &memReq);
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            bool typeOk = (memReq.memoryTypeBits & (1u << i)) != 0;
            bool flagsOk = (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            if (typeOk && flagsOk)
            {
                memoryTypeIndex = i;
                break;
            }
        }
        if (memoryTypeIndex == UINT32_MAX)
            throw std::runtime_error("Showcase: no suitable memory type for offscreen image!");
        
        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memReq.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(device, &allocateInfo, nullptr, &offscreenMemory[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to allocate offscreen image memory!");
    
        vkBindImageMemory(device, offscreenImage[i], offscreenMemory[i], 0);

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = offscreenImage[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = createInfo.format;
        viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;
    
        if (vkCreateImageView(device, &viewCreateInfo, nullptr, &offscreenView[i]) != VK_SUCCESS)
        {
            if (offscreenImage[i])  { vkDestroyImage(device, offscreenImage[i], nullptr);  offscreenImage[i] = VK_NULL_HANDLE; }
            if (offscreenMemory[i]) { vkFreeMemory(device, offscreenMemory[i], nullptr);   offscreenMemory[i] = VK_NULL_HANDLE; }
            throw std::runtime_error("Showcase: failed to create offscreen image view!");
        }
        VkSemaphoreCreateInfo semCreateInfo{};
        semCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semCreateInfo, nullptr, &computeDone[i]) != VK_SUCCESS)
        {
            if (offscreenView[i])   { vkDestroyImageView(device, offscreenView[i], nullptr); offscreenView[i] = VK_NULL_HANDLE; }
            if (offscreenImage[i])  { vkDestroyImage(device, offscreenImage[i], nullptr);    offscreenImage[i] = VK_NULL_HANDLE; }
            if (offscreenMemory[i]) { vkFreeMemory(device, offscreenMemory[i], nullptr);     offscreenMemory[i] = VK_NULL_HANDLE; }
            throw std::runtime_error("Showcase: failed to create compute semaphore");
        }
        offscreenValid[i] = true;
    }
}

void ShowcaseApp::createFSRTargets()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        fsrValid[i] = false;
        VkExtent3D extent3D{};
        extent3D.width  = swapChainExtent.width;
        extent3D.height = swapChainExtent.height;
        extent3D.depth  = 1;

        VkImageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        createInfo.imageType = VK_IMAGE_TYPE_2D;
        createInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        createInfo.extent = extent3D;
        createInfo.mipLevels = 1;
        createInfo.arrayLayers = 1;
        createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        createInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        
        uint32_t fams[2] = { graphicsFamily, computeFamily };
        uint32_t unique[2];
        uint32_t uniqueCount = 0;

        for (uint32_t j = 0; j < 2; ++j)
        {
            uint32_t f = fams[j];
            bool seen = false;
            for (uint32_t k = 0; k < uniqueCount; ++k)
            {
                if (unique[k] == f) { seen = true; break; }
            }
            if (!seen)
                unique[uniqueCount++] = f;
        }

        if (uniqueCount > 1)
        {
            createInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = uniqueCount;
            createInfo.pQueueFamilyIndices   = unique;
        }
        else
        {
            createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices   = nullptr;
        }

        if (vkCreateImage(device, &createInfo, nullptr, &fsrImage[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create offscreen image!");
        {
            char name[64];
            std::snprintf(name, sizeof(name), "FSRImage[%u]", i);
            setImageName(fsrImage[i], name);
        }
        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(device, fsrImage[i], &memReq);
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            bool typeOk = (memReq.memoryTypeBits & (1u << i)) != 0;
            bool flagsOk = (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            if (typeOk && flagsOk)
            {
                memoryTypeIndex = i;
                break;
            }
        }
        if (memoryTypeIndex == UINT32_MAX)
            throw std::runtime_error("Showcase: no suitable memory type for offscreen image!");
        
        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memReq.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(device, &allocateInfo, nullptr, &fsrMemory[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to allocate offscreen image memory!");
    
        vkBindImageMemory(device, fsrImage[i], fsrMemory[i], 0);

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = fsrImage[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = createInfo.format;
        viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;
    
        if (vkCreateImageView(device, &viewCreateInfo, nullptr, &fsrView[i]) != VK_SUCCESS)
        {
            if (fsrImage[i])  { vkDestroyImage(device, fsrImage[i], nullptr);  fsrImage[i] = VK_NULL_HANDLE; }
            if (fsrMemory[i]) { vkFreeMemory(device, fsrMemory[i], nullptr);   fsrMemory[i] = VK_NULL_HANDLE; }
            throw std::runtime_error("Showcase: failed to create fsr image view!");
        }
        fsrValid[i] = true;
    }
}

void ShowcaseApp::createNrdTargets()
{
    for (uint32_t idx = 0; idx < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++idx)
    {
        nrdOutputsInitialized[idx] = false;
        nrdOutputsSampled[idx]     = false;
    }

    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        auto& frame = nrdFrameImages[i];

        // Destroy old stuff if this is a resize / re-init
        if (frame.diffView)   { vkDestroyImageView(device, frame.diffView,  nullptr); frame.diffView  = VK_NULL_HANDLE; }
        if (frame.diffImage)  { vkDestroyImage(device,     frame.diffImage, nullptr); frame.diffImage = VK_NULL_HANDLE; }
        if (frame.diffMemory) { vkFreeMemory(device,       frame.diffMemory,nullptr); frame.diffMemory= VK_NULL_HANDLE; }

        if (frame.specView)   { vkDestroyImageView(device, frame.specView,  nullptr); frame.specView  = VK_NULL_HANDLE; }
        if (frame.specImage)  { vkDestroyImage(device,     frame.specImage, nullptr); frame.specImage = VK_NULL_HANDLE; }
        if (frame.specMemory) { vkFreeMemory(device,       frame.specMemory,nullptr); frame.specMemory= VK_NULL_HANDLE; }

        frame.valid = false;

        VkExtent3D extent3D{};
        extent3D.width  = rayTraceExtent.width;
        extent3D.height = rayTraceExtent.height;
        extent3D.depth  = 1;

        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = VK_FORMAT_R16G16B16A16_SFLOAT;   // RGBA16F: radiance + hitdist
        ci.extent        = extent3D;
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // Queue sharing like before (graphics + compute)
        uint32_t fams[2]       = { graphicsFamily, computeFamily };
        uint32_t unique[2];
        uint32_t uniqueCount   = 0;
        for (uint32_t j = 0; j < 2; ++j)
        {
            uint32_t f = fams[j];
            bool seen = false;
            for (uint32_t k = 0; k < uniqueCount; ++k)
            {
                if (unique[k] == f) { seen = true; break; }
            }
            if (!seen)
                unique[uniqueCount++] = f;
        }

        if (uniqueCount > 1)
        {
            ci.sharingMode           = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = uniqueCount;
            ci.pQueueFamilyIndices   = unique;
        }
        else
        {
            ci.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
            ci.queueFamilyIndexCount = 0;
            ci.pQueueFamilyIndices   = nullptr;
        }

        auto allocateImage = [&](VkImage& image, VkDeviceMemory& memory, VkImageView& view, const char* debugName)
        {
            // Create image
            if (vkCreateImage(device, &ci, nullptr, &image) != VK_SUCCESS)
                throw std::runtime_error("Showcase: failed to create NRD output image!");
            {
                setImageName(image, debugName);
            }

            VkMemoryRequirements memReq{};
            vkGetImageMemoryRequirements(device, image, &memReq);

            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

            uint32_t memoryTypeIndex = UINT32_MAX;
            for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m)
            {
                bool typeOk  = (memReq.memoryTypeBits & (1u << m)) != 0;
                bool flagsOk = (memProps.memoryTypes[m].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                if (typeOk && flagsOk)
                {
                    memoryTypeIndex = m;
                    break;
                }
            }
            if (memoryTypeIndex == UINT32_MAX)
                throw std::runtime_error("Showcase: no suitable memory type for NRD output image!");

            VkMemoryAllocateInfo mai{};
            mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize  = memReq.size;
            mai.memoryTypeIndex = memoryTypeIndex;

            if (vkAllocateMemory(device, &mai, nullptr, &memory) != VK_SUCCESS)
                throw std::runtime_error("Showcase: failed to allocate NRD output image memory!");

            vkBindImageMemory(device, image, memory, 0);

            // View
            VkImageViewCreateInfo viewCI{};
            viewCI.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewCI.image    = image;
            viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewCI.format   = ci.format;
            viewCI.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY
            };
            viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            viewCI.subresourceRange.baseMipLevel   = 0;
            viewCI.subresourceRange.levelCount     = 1;
            viewCI.subresourceRange.baseArrayLayer = 0;
            viewCI.subresourceRange.layerCount     = 1;

            if (vkCreateImageView(device, &viewCI, nullptr, &view) != VK_SUCCESS)
            {
                vkDestroyImage(device, image, nullptr);
                vkFreeMemory(device, memory, nullptr);
                image  = VK_NULL_HANDLE;
                memory = VK_NULL_HANDLE;
                throw std::runtime_error("Showcase: failed to create NRD output image view!");
            }
        };

        // Create diffuse + spec images
        char diffName[64];
        std::snprintf(diffName, sizeof(diffName), "NRD_OUT_Diff[%u]", i);
        char specName[64];
        std::snprintf(specName, sizeof(specName), "NRD_OUT_Spec[%u]", i);
        allocateImage(frame.diffImage, frame.diffMemory, frame.diffView, diffName);
        allocateImage(frame.specImage, frame.specMemory, frame.specView, specName);

        frame.valid = true;
    }
}


void ShowcaseApp::initNRD()
{
    nrd::IntegrationCreationDesc integDesc{};
    std::strncpy(integDesc.name, "NRD", 3);

    integDesc.resourceWidth  = static_cast<uint16_t>(rayTraceExtent.width);
    integDesc.resourceHeight = static_cast<uint16_t>(rayTraceExtent.height);

    
    integDesc.queuedFrameNum = static_cast<uint8_t>(SwapChain::MAX_FRAMES_IN_FLIGHT);

    
    integDesc.enableWholeLifetimeDescriptorCaching = true;
    integDesc.autoWaitForIdle = true;
    integDesc.demoteFloat32to16 = false;
    integDesc.promoteFloat16to32 = true;

    static const nrd::DenoiserDesc denoiserDescs[] =
    {
        { nrdRelaxId, nrd::Denoiser::RELAX_DIFFUSE_SPECULAR }
    };
    nrd::InstanceCreationDesc instCreateDesc{};
    instCreateDesc.denoisersNum = 1;
    instCreateDesc.denoisers    = denoiserDescs;
    
    nri::DeviceCreationVKDesc deviceVKDesc{};
    const nrd::LibraryDesc* libDesc = nrd::GetLibraryDesc();
    deviceVKDesc.vkBindingOffsets.samplerOffset               = libDesc->spirvBindingOffsets.samplerOffset;
    deviceVKDesc.vkBindingOffsets.textureOffset               = libDesc->spirvBindingOffsets.textureOffset;
    deviceVKDesc.vkBindingOffsets.constantBufferOffset        = libDesc->spirvBindingOffsets.constantBufferOffset;
    deviceVKDesc.vkBindingOffsets.storageTextureAndBufferOffset =
        libDesc->spirvBindingOffsets.storageTextureAndBufferOffset;

    // --- Extensions: tell NRI what we actually enabled on the Vulkan device ---
    deviceVKDesc.vkExtensions.deviceExtensions      = deviceExtensions.data();
    deviceVKDesc.vkExtensions.deviceExtensionNum    = static_cast<uint32_t>(deviceExtensions.size());
    deviceVKDesc.vkInstance       = instance;
    deviceVKDesc.vkPhysicalDevice = physicalDevice;
    deviceVKDesc.vkDevice         = device;
    std::array<nri::QueueFamilyVKDesc, 2> queueFamilies{};
    
    queueFamilies[0].queueNum    = 1;
    queueFamilies[0].queueType   = nri::QueueType::GRAPHICS;
    queueFamilies[0].familyIndex = graphicsFamily;

    queueFamilies[1].queueNum    = 1;
    queueFamilies[1].queueType   = nri::QueueType::COMPUTE;
    queueFamilies[1].familyIndex = computeFamily;

    deviceVKDesc.queueFamilies  = queueFamilies.data();
    deviceVKDesc.queueFamilyNum = 2;

    deviceVKDesc.minorVersion = 3;

    deviceVKDesc.enableNRIValidation            = true;
    deviceVKDesc.enableMemoryZeroInitialization = false;

    nrd::Result r = nrdIntegration.RecreateVK(integDesc, instCreateDesc, deviceVKDesc);
    if (r != nrd::Result::SUCCESS)
        throw std::runtime_error("Showcase: Failed to create NRD Integration (RecreateVK)");
}

void ShowcaseApp::createNrdInputTargets()
{
    uint32_t w = rayTraceExtent.width;
    uint32_t h = rayTraceExtent.height;

    for (uint32_t idx = 0; idx < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++idx)
    {
        nrdInputsInitialized[idx] = false;
        nrdInputsSampled[idx]     = false;
    }

    auto destroyTex = [&](NrdTexture& t)
    {
        if (t.view)   { vkDestroyImageView(device, t.view, nullptr);  t.view = VK_NULL_HANDLE; }
        if (t.image)  { vkDestroyImage(device, t.image, nullptr);     t.image = VK_NULL_HANDLE; }
        if (t.memory) { vkFreeMemory(device, t.memory, nullptr);      t.memory = VK_NULL_HANDLE; }
        t.format = VK_FORMAT_UNDEFINED;
    };

    // Destroy old if any
    for (auto& frame : nrdInputs)
    {
        destroyTex(frame.diffRadianceHit);
        destroyTex(frame.specRadianceHit);
        destroyTex(frame.normalRoughness);
        destroyTex(frame.viewZ);
        destroyTex(frame.motionVec);
        frame.valid = false;
    }

    // Helper: create and allocate a 2D texture for NRD input
    auto createTex = [&](VkFormat fmt, uint32_t w, uint32_t h, NrdTexture& out, const char* debugName)
    {
        out.width  = w;
        out.height = h;
        out.format = fmt;

        VkImageCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType     = VK_IMAGE_TYPE_2D;
        ci.format        = fmt;
        ci.extent        = { w, h, 1 };
        ci.mipLevels     = 1;
        ci.arrayLayers   = 1;
        ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | 
                           VK_IMAGE_USAGE_SAMPLED_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        uint32_t fams[2] = { graphicsFamily, computeFamily };
        uint32_t unique[2];
        uint32_t uniqueCount = 0;
        for (uint32_t j = 0; j < 2; ++j)
        {
            uint32_t f = fams[j];
            bool seen = false;
            for (uint32_t k = 0; k < uniqueCount; ++k)
            {
                if (unique[k] == f) { seen = true; break; }
            }
            if (!seen)
                unique[uniqueCount++] = f;
        }

        if (uniqueCount > 1)
        {
            ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = uniqueCount;
            ci.pQueueFamilyIndices = unique;
        }
        else
        {
            ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.queueFamilyIndexCount = 0;
            ci.pQueueFamilyIndices = nullptr;
        }

        if (vkCreateImage(device, &ci, nullptr, &out.image) != VK_SUCCESS)
            throw std::runtime_error("NRD: failed to create input image!");
        setImageName(out.image, debugName);

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, out.image, &req);

        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            bool typeOk = (req.memoryTypeBits & (1u << i)) != 0;
            bool flagsOk = (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (typeOk && flagsOk)
            {
                memoryTypeIndex = i;
                break;
            }
        }
        if (memoryTypeIndex == UINT32_MAX)
            throw std::runtime_error("NRD: no suitable memory type for input image!");

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(device, &mai, nullptr, &out.memory) != VK_SUCCESS)
            throw std::runtime_error("NRD: failed to allocate memory for input image!");

        vkBindImageMemory(device, out.image, out.memory, 0);

        VkImageViewCreateInfo viewCI{};
        viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.image = out.image;
        viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format = fmt;
        viewCI.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY
        };
        viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.baseMipLevel = 0;
        viewCI.subresourceRange.levelCount = 1;
        viewCI.subresourceRange.baseArrayLayer = 0;
        viewCI.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewCI, nullptr, &out.view) != VK_SUCCESS)
            throw std::runtime_error("NRD: failed to create input image view!");
    };

    for (uint32_t idx = 0; idx < nrdInputs.size(); ++idx)
    {
        auto& frame = nrdInputs[idx];
        char diffName[64];   std::snprintf(diffName,   sizeof(diffName),   "NRD_IN_Diff[%u]", idx);
        char specName[64];   std::snprintf(specName,   sizeof(specName),   "NRD_IN_Spec[%u]", idx);
        char nrName[64];     std::snprintf(nrName,     sizeof(nrName),     "NRD_IN_NormalRough[%u]", idx);
        char viewZName[64];  std::snprintf(viewZName,  sizeof(viewZName),  "NRD_IN_ViewZ[%u]", idx);
        char mvName[64];     std::snprintf(mvName,     sizeof(mvName),     "NRD_IN_Motion[%u]", idx);

        createTex(VK_FORMAT_R16G16B16A16_SFLOAT,           w, h, frame.diffRadianceHit, diffName);
        createTex(VK_FORMAT_R16G16B16A16_SFLOAT,           w, h, frame.specRadianceHit, specName);
        createTex(VK_FORMAT_R16G16B16A16_SNORM,            w, h, frame.normalRoughness, nrName);
        createTex(VK_FORMAT_R32_SFLOAT,                    w, h, frame.viewZ,            viewZName);
        createTex(VK_FORMAT_R16G16_SNORM,                  w, h, frame.motionVec,        mvName);

        frame.valid = true;
    }
}


void ShowcaseApp::createComputeDescriptors()
{
    // ---------- SET 0: STATIC SCENE DATA ----------
    // bindings:
    //   0: sbvhNodesBuffer
    //   1: triangleBuffer
    //   2: shadingBuffer
    //   3: materialBuffer
    //   4: emissiveTriangleBuffer
    //   5: lightProbBuffer
    //   6: lightPdfBuffer
    //   7: lightAliasBuffer
    //   8: triToLightIdxBuffer
    //   9: textures[]

    VkDescriptorSetLayoutBinding set0[10]{};

    auto initSet0 = [](VkDescriptorSetLayoutBinding& b, uint32_t binding,
                     VkDescriptorType type, uint32_t count, VkShaderStageFlags stages)
    {
        b = {};
        b.binding         = binding;
        b.descriptorType  = type;
        b.descriptorCount = count;
        b.stageFlags      = stages;
    };

    initSet0(set0[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[6], 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[7], 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    initSet0(set0[8], 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    uint32_t maxTextures = std::max(1u, static_cast<uint32_t>(flattened.size()));
    initSet0(set0[9], 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  maxTextures, VK_SHADER_STAGE_COMPUTE_BIT);

    VkDescriptorSetLayoutCreateInfo staticLayoutCreateInfo{ };
    staticLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    staticLayoutCreateInfo.bindingCount = 10;
    staticLayoutCreateInfo.pBindings    = set0;

    if (vkCreateDescriptorSetLayout(device, &staticLayoutCreateInfo, nullptr, &computeStaticSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute static descriptor set layout!");


    // ---------- SET 1: PER-FRAME DATA ----------
    // simple path: only TLAS + params + offscreen
    // wavefront path: includes FSR + NRD resources
    std::vector<VkDescriptorSetLayoutBinding> set1;
    auto initSet1 = [](VkDescriptorSetLayoutBinding& b, uint32_t binding,
                     VkDescriptorType type, VkShaderStageFlags stages)
    {
        b = {};
        b.binding         = binding;
        b.descriptorType  = type;
        b.descriptorCount = 1;
        b.stageFlags      = stages;
    };

    if constexpr (SimpleRayTrace)
    {
        set1.resize(5);
        initSet1(set1[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // TLAS nodes
        initSet1(set1[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // TLAS instances
        initSet1(set1[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // TLAS indices
        initSet1(set1[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // params
        initSet1(set1[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT); // offscreen image
    }
    else
    {
        set1.resize(16);
        initSet1(set1[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // TLAS nodes
        initSet1(set1[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // TLAS instances
        initSet1(set1[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // TLAS indices
        initSet1(set1[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // params
        initSet1(set1[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT); // offscreen image
        initSet1(set1[5], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  VK_SHADER_STAGE_COMPUTE_BIT); // fsrImage
        initSet1(set1[6], 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT); // FSR UBO
        initSet1(set1[7], 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT);
        initSet1(set1[8], 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT);
        initSet1(set1[9],  9,  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT); // diff radiance + hitdist
        initSet1(set1[10], 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT); // spec radiance + hitdist
        initSet1(set1[11], 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT); // normal + roughness
        initSet1(set1[12], 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT); // viewZ
        initSet1(set1[13], 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT); // motion vectors
        initSet1(set1[14], 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);
        initSet1(set1[15], 15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    VkDescriptorSetLayoutCreateInfo frameLayoutCreateInfo{ };
    frameLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    frameLayoutCreateInfo.bindingCount = static_cast<uint32_t>(set1.size());
    frameLayoutCreateInfo.pBindings    = set1.data();

    if (vkCreateDescriptorSetLayout(device, &frameLayoutCreateInfo, nullptr, &computeFrameSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute frame descriptor set layout!");


    // ---------- SET 2: DYNAMIC (WAVEFRONT) ----------
    // bindings:
    //   0: PathHeader[]
    //   1: Ray[]
    //   2: HitIds[]
    //   3: Hitdata[]
    //   4: RadianceState[]
    //   5: BsdfSample[]
    //   6: LightSample[]
    //   7: ShadowRay[]
    //   8: ShadowResult[]
    //   9: PathQueue[]
    //  10: PixelStats[]
    //  11: HighVarPixelIndices[]
    //  12: AdaptiveCounters

    VkDescriptorSetLayoutBinding d[13]{};

    auto initSet2 = [](VkDescriptorSetLayoutBinding& b, uint32_t binding)
    {
        b = {};
        b.binding         = binding;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    };

    for (uint32_t b = 0; b < 13; ++b)
        initSet2(d[b], b);

    VkDescriptorSetLayoutCreateInfo dynamicLayoutCreateInfo{ };
    dynamicLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dynamicLayoutCreateInfo.bindingCount = 13;
    dynamicLayoutCreateInfo.pBindings    = d;

    if (vkCreateDescriptorSetLayout(device, &dynamicLayoutCreateInfo, nullptr, &computeDynamicSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute dynamic descriptor set layout!");


    VkDescriptorPoolSize poolSizes[4]{};

    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 9 * SwapChain::MAX_FRAMES_IN_FLIGHT;

    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 27 * SwapChain::MAX_FRAMES_IN_FLIGHT;

    poolSizes[2].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = maxTextures + 2 * SwapChain::MAX_FRAMES_IN_FLIGHT;

    poolSizes[3].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[3].descriptorCount = SwapChain::MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolSize uboPool{};
    uboPool.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboPool.descriptorCount = SwapChain::MAX_FRAMES_IN_FLIGHT;

    
    VkDescriptorPoolCreateInfo poolCreateInfo{ };
    poolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 4;
    poolCreateInfo.pPoolSizes    = poolSizes;
    poolCreateInfo.maxSets       = 1 + 2 * SwapChain::MAX_FRAMES_IN_FLIGHT; // 1 static + N frame + N dynamic

    if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &computeDescPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute descriptor pool!");



    VkDescriptorSetLayout staticLayout = computeStaticSetLayout;
    VkDescriptorSetAllocateInfo allocateInfo0{ };
    allocateInfo0.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo0.descriptorPool     = computeDescPool;
    allocateInfo0.descriptorSetCount = 1;
    allocateInfo0.pSetLayouts        = &staticLayout;

    if (vkAllocateDescriptorSets(device, &allocateInfo0, &computeStaticSet) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate compute static descriptor set!");
    setDescriptorSetName(computeStaticSet, "ComputeStaticSet");



    std::array<VkDescriptorSetLayout, SwapChain::MAX_FRAMES_IN_FLIGHT> frameLayouts{};
    frameLayouts.fill(computeFrameSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo1{ };
    allocateInfo1.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo1.descriptorPool     = computeDescPool;
    allocateInfo1.descriptorSetCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    allocateInfo1.pSetLayouts        = frameLayouts.data();

    if (vkAllocateDescriptorSets(device, &allocateInfo1, computeFrameSets) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate compute frame descriptor sets!");
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "ComputeFrameSet[%u]", i);
        setDescriptorSetName(computeFrameSets[i], name);
    }



    std::array<VkDescriptorSetLayout, SwapChain::MAX_FRAMES_IN_FLIGHT> dynLayouts{};
    dynLayouts.fill(computeDynamicSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo2{ };
    allocateInfo2.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo2.descriptorPool     = computeDescPool;
    allocateInfo2.descriptorSetCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    allocateInfo2.pSetLayouts        = dynLayouts.data();

    if (vkAllocateDescriptorSets(device, &allocateInfo2, computeDynamicSets) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate compute dynamic descriptor sets!");
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "ComputeDynamicSet[%u]", i);
        setDescriptorSetName(computeDynamicSets[i], name);
    }
}


void ShowcaseApp::updateComputeDescriptor(int frameIndex)
{
    auto updateForFrame = [&](uint32_t i)
    {
        VkDescriptorImageInfo offscreenInfo{};
        offscreenInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        offscreenInfo.imageView   = offscreenView[i];
        offscreenInfo.sampler     = VK_NULL_HANDLE;

        VkDescriptorBufferInfo paramsInfo{};
        paramsInfo.buffer = paramsBuffer[i];
        paramsInfo.offset = 0;
        paramsInfo.range  = sizeof(ParamsGPU);

        VkDescriptorBufferInfo tlasNodesInfo{ tlasNodesBuf[i], 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo tlasInstInfo { tlasInstBuf[i],  0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo tlasIdxInfo  { tlasIdxBuf[i],   0, VK_WHOLE_SIZE };

        if constexpr (SimpleRayTrace)
        {
            // Skip until TLAS/params/offscreen are ready to avoid null descriptor writes.
            if (tlasNodesBuf[i] == VK_NULL_HANDLE ||
                tlasInstBuf[i]  == VK_NULL_HANDLE ||
                tlasIdxBuf[i]   == VK_NULL_HANDLE ||
                paramsBuffer[i] == VK_NULL_HANDLE ||
                offscreenView[i] == VK_NULL_HANDLE)
            {
                return;
            }

            VkWriteDescriptorSet writes[5]{};
            auto init = [&](VkWriteDescriptorSet& w, uint32_t binding, VkDescriptorType type, const void* info)
            {
                w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = computeFrameSets[i];
                w.dstBinding = binding;
                w.descriptorType = type;
                w.descriptorCount = 1;
                if (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    w.pBufferInfo = reinterpret_cast<const VkDescriptorBufferInfo*>(info);
                else
                    w.pImageInfo = reinterpret_cast<const VkDescriptorImageInfo*>(info);
            };

            init(writes[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &tlasNodesInfo);
            init(writes[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &tlasInstInfo);
            init(writes[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &tlasIdxInfo);
            init(writes[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &paramsInfo);
            init(writes[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &offscreenInfo);
            vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
        }
        else
        {
            VkDescriptorImageInfo fsrInfo{};
            fsrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            fsrInfo.imageView   = fsrView[i];
            fsrInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorBufferInfo fsrBuf{};
            fsrBuf.buffer = fsrConstBuffer[i];
            fsrBuf.offset = 0;
            fsrBuf.range  = sizeof(FsrConstants);

            VkDescriptorImageInfo fsrInputSamplerInfo{};
            fsrInputSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fsrInputSamplerInfo.imageView   = offscreenView[i];
            fsrInputSamplerInfo.sampler     = offscreenSampler;

            VkDescriptorImageInfo rcasInput{};
            rcasInput.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            rcasInput.imageView   = fsrView[i];
            rcasInput.sampler     = offscreenSampler;

            VkDescriptorImageInfo nrdDiffInfo{};
            nrdDiffInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            nrdDiffInfo.imageView   = nrdFrameImages[i].diffView;
            nrdDiffInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorImageInfo nrdSpecInfo{};
            nrdSpecInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            nrdSpecInfo.imageView   = nrdFrameImages[i].specView;
            nrdSpecInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorImageInfo inDiffInfo{};
            inDiffInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            inDiffInfo.imageView   = nrdInputs[i].diffRadianceHit.view;
            inDiffInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorImageInfo inSpecInfo{};
            inSpecInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            inSpecInfo.imageView   = nrdInputs[i].specRadianceHit.view;
            inSpecInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorImageInfo inNRInfo{};
            inNRInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            inNRInfo.imageView   = nrdInputs[i].normalRoughness.view;
            inNRInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorImageInfo inViewZInfo{};
            inViewZInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            inViewZInfo.imageView   = nrdInputs[i].viewZ.view;
            inViewZInfo.sampler     = VK_NULL_HANDLE;

            VkDescriptorImageInfo inMotionInfo{};
            inMotionInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            inMotionInfo.imageView   = nrdInputs[i].motionVec.view;
            inMotionInfo.sampler     = VK_NULL_HANDLE;

            VkWriteDescriptorSet writes[12]{};

            // 4: offscreen image (write-only for main pathtrace / combine)
            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = computeFrameSets[i];
            writes[0].dstBinding      = 4;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[0].descriptorCount = 1;
            writes[0].pImageInfo      = &offscreenInfo;

            // 5: FSR output storage image
            writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet          = computeFrameSets[i];
            writes[1].dstBinding      = 5;
            writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].descriptorCount = 1;
            writes[1].pImageInfo      = &fsrInfo;

            // 6: FSR constant buffer
            writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet          = computeFrameSets[i];
            writes[2].dstBinding      = 6;
            writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[2].descriptorCount = 1;
            writes[2].pBufferInfo     = &fsrBuf;

            // 7: FSR input sampler (denoised + combined image)
            writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet          = computeFrameSets[i];
            writes[3].dstBinding      = 7;
            writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].descriptorCount = 1;
            writes[3].pImageInfo      = &fsrInputSamplerInfo;

            // 8: RCAS input sampler
            writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet          = computeFrameSets[i];
            writes[4].dstBinding      = 8;
            writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[4].descriptorCount = 1;
            writes[4].pImageInfo      = &rcasInput;

            // 14: NRD OUT_DIFF_RADIANCE_HITDIST
            writes[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[5].dstSet          = computeFrameSets[i];
            writes[5].dstBinding      = 14;
            writes[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[5].descriptorCount = 1;
            writes[5].pImageInfo      = &nrdDiffInfo;

            // 15: NRD OUT_SPEC_RADIANCE_HITDIST
            writes[6].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[6].dstSet          = computeFrameSets[i];
            writes[6].dstBinding      = 15;
            writes[6].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[6].descriptorCount = 1;
            writes[6].pImageInfo      = &nrdSpecInfo;

            // 9: IN_DIFF_RADIANCE_HITDIST
            writes[7].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[7].dstSet          = computeFrameSets[i];
            writes[7].dstBinding      = 9;
            writes[7].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[7].descriptorCount = 1;
            writes[7].pImageInfo      = &inDiffInfo;
                
            // 10: IN_SPEC_RADIANCE_HITDIST
            writes[8].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[8].dstSet          = computeFrameSets[i];
            writes[8].dstBinding      = 10;
            writes[8].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[8].descriptorCount = 1;
            writes[8].pImageInfo      = &inSpecInfo;
                
            // 11: IN_NORMAL_ROUGHNESS
            writes[9].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[9].dstSet          = computeFrameSets[i];
            writes[9].dstBinding      = 11;
            writes[9].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[9].descriptorCount = 1;
            writes[9].pImageInfo      = &inNRInfo;
                
            // 12: IN_VIEWZ
            writes[10].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[10].dstSet          = computeFrameSets[i];
            writes[10].dstBinding      = 12;
            writes[10].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[10].descriptorCount = 1;
            writes[10].pImageInfo      = &inViewZInfo;
                
            // 13: IN_MV
            writes[11].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[11].dstSet          = computeFrameSets[i];
            writes[11].dstBinding      = 13;
            writes[11].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[11].descriptorCount = 1;
            writes[11].pImageInfo      = &inMotionInfo;
                
            vkUpdateDescriptorSets(device, 12, writes, 0, nullptr);
        }
    };

    if (frameIndex >= 0)
    {
        updateForFrame(static_cast<uint32_t>(frameIndex));
    }
    else
    {
        for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
            updateForFrame(i);
    }
}



void ShowcaseApp::writeStaticComputeBindings()
{
    VkDescriptorBufferInfo sbvhInfo{ sbvhNodesBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo triangleInfo{ triangleBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo shadeInfo{ shadingBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo matInfo{ materialBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo eTriInfo{ emissiveTriangleBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo lightProbInfo{ lightProbBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo lightPdfInfo{ lightPdfBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo lightAliasInfo{ lightAliasBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo triToLightIdxInfo{ triToLightIdxBuffer, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[9]{};

    auto initWrite = [&](VkWriteDescriptorSet& w, uint32_t binding, const VkDescriptorBufferInfo* info)
    {
        w = {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = computeStaticSet;
        w.dstBinding      = binding;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = info;
    };

    initWrite(writes[0], 0, &sbvhInfo);
    initWrite(writes[1], 1, &triangleInfo);
    initWrite(writes[2], 2, &shadeInfo);
    initWrite(writes[3], 3, &matInfo);
    if constexpr (!SimpleRayTrace)
    {
        initWrite(writes[4], 4, &eTriInfo);
        initWrite(writes[5], 5, &lightProbInfo);
        initWrite(writes[6], 6, &lightPdfInfo);
        initWrite(writes[7], 7, &lightAliasInfo);
        initWrite(writes[8], 8, &triToLightIdxInfo);
        vkUpdateDescriptorSets(device, 9, writes, 0, nullptr);
    }
    else
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

}


void ShowcaseApp::uploadStaticData()
{
    uploadDeviceLocal(SBVHNodes, 0, sbvhNodesBuffer, sbvhNodesMemory);
    uploadDeviceLocal(intersectionTrinagles, 0, triangleBuffer, triangleMemory);
    uploadDeviceLocal(shadingTriangles, 0, shadingBuffer, shadingMemory);
    uploadDeviceLocal(materials, 0, materialBuffer, materialMemory);
    if constexpr (!SimpleRayTrace)
    {
        uploadDeviceLocal(emissiveTrinagles, 0, emissiveTriangleBuffer, emissiveTriangleMemory);
        uploadDeviceLocal(lightProb, 0, lightProbBuffer, lightProbMemory);
        uploadDeviceLocal(lightPdf, 0, lightPdfBuffer, lightPdfMemory);
        uploadDeviceLocal(lightAlias, 0, lightAliasBuffer, lightAliasMemory);
        uploadDeviceLocal(triToLightIdx, 0, triToLightIdxBuffer, triToLightIdxMemory);
    }
    uploadTextureImages();
    writeStaticComputeBindings();
}

void ShowcaseApp::writeParamsBindingForFrame(uint32_t frameIndex)
{
    VkDescriptorBufferInfo paramsInfo{ paramsBuffer[frameIndex], 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = computeFrameSets[frameIndex];
    w.dstBinding      = 3;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.descriptorCount = 1;
    w.pBufferInfo     = &paramsInfo;

    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

void ShowcaseApp::createParamsBuffers()
{
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        createBuffer(device, physicalDevice, sizeof(ParamsGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            paramsBuffer[i], paramsMemory[i]);

        VkResult result = vkMapMemory(device, paramsMemory[i], 0, sizeof(ParamsGPU), 0, &paramsMapped[i]);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Map params buffer failed");
    }
}

void ShowcaseApp::createFsrConstBuffers()
{
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        createBuffer(device, physicalDevice, sizeof(FsrConstants),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        fsrConstBuffer[i], fsrConstMemory[i]);
        
        VkResult result = vkMapMemory(device, fsrConstMemory[i], 0, sizeof(FsrConstants), 0, &fsrConstMapped[i]);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Map FSR buffer failed");
    }
}



void ShowcaseApp::createComputePipeline()
{
    if (computePipeline)
    {
        vkDestroyPipeline(device, computePipeline, nullptr);
        computePipeline = VK_NULL_HANDLE;
    }

    if (rayTraceLogicPipeline)
    {
        vkDestroyPipeline(device, rayTraceLogicPipeline, nullptr);
        rayTraceLogicPipeline = VK_NULL_HANDLE;
    }
    if (rayTraceNewPathPipeline)
    {
        vkDestroyPipeline(device, rayTraceNewPathPipeline, nullptr);
        rayTraceNewPathPipeline = VK_NULL_HANDLE;
    }
    if (rayTraceMaterialPipeline)
    {
        vkDestroyPipeline(device, rayTraceMaterialPipeline, nullptr);
        rayTraceMaterialPipeline = VK_NULL_HANDLE;
    }
    if (rayTraceExtendRayPipeline)
    {
        vkDestroyPipeline(device, rayTraceExtendRayPipeline, nullptr);
        rayTraceExtendRayPipeline = VK_NULL_HANDLE;
    }
    if (rayTraceShadowRayPipeline)
    {
        vkDestroyPipeline(device, rayTraceShadowRayPipeline, nullptr);
        rayTraceShadowRayPipeline = VK_NULL_HANDLE;
    }
    if (rayTraceFinalWritePipeline)
    {
        vkDestroyPipeline(device, rayTraceFinalWritePipeline, nullptr);
        rayTraceFinalWritePipeline = VK_NULL_HANDLE;
    }
    if (FSRPipeline)
    {
        vkDestroyPipeline(device, FSRPipeline, nullptr);
        FSRPipeline = VK_NULL_HANDLE;
    }
    if (FSRSharpenPipeline)
    {
        vkDestroyPipeline(device, FSRSharpenPipeline, nullptr);
        FSRSharpenPipeline = VK_NULL_HANDLE;
    }

    if (computePipelineLayout)
    {
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }

    // --- Pipeline layout: 3 descriptor sets (static, per-frame, dynamic) ---

    VkDescriptorSetLayout setLayouts[3] = {
        computeStaticSetLayout,  // set 0: static scene + textures
        computeFrameSetLayout,   // set 1: TLAS + params
        computeDynamicSetLayout  // set 2: wavefront buffers
    };

    VkPipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = 3;
    layoutCreateInfo.pSetLayouts    = setLayouts;
    layoutCreateInfo.pushConstantRangeCount = 0;
    layoutCreateInfo.pPushConstantRanges    = nullptr;

    if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &computePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute pipeline layout!");

    // --- Helper to create a compute pipeline from a SPIR-V file ---

    auto makeComputePipeline = [&](const char* path, VkPipeline& outPipeline)
    {
        auto code = Pipeline::readFile(path);
        VkShaderModule shader = createShaderModule(code);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName  = "main";

        VkComputePipelineCreateInfo createInfo{};
        createInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        createInfo.stage  = stage;
        createInfo.layout = computePipelineLayout;
        createInfo.basePipelineHandle = VK_NULL_HANDLE;
        createInfo.basePipelineIndex  = -1;

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &outPipeline) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device, shader, nullptr);
            throw std::runtime_error(std::string("Showcase: failed to create compute pipeline for ") + path);
        }

        vkDestroyShaderModule(device, shader, nullptr);
    };

    if constexpr (SimpleRayTrace)
    {
        // Simple path: single mega kernel like before
        makeComputePipeline("build/shaders/rayTraceSimple.comp.spv", computePipeline);
    }
    else
    {
        // Wavefront path: multiple kernels
        makeComputePipeline("build/shaders/rayTraceLogic.comp.spv",           rayTraceLogicPipeline);
        makeComputePipeline("build/shaders/rayTraceNewPath.comp.spv",       rayTraceNewPathPipeline);
        makeComputePipeline("build/shaders/rayTraceMaterial.comp.spv",     rayTraceMaterialPipeline);
        makeComputePipeline("build/shaders/rayTraceExtendRay.comp.spv",   rayTraceExtendRayPipeline);
        makeComputePipeline("build/shaders/rayTraceShadowRay.comp.spv",   rayTraceShadowRayPipeline);
        makeComputePipeline("build/shaders/rayTraceFinalWrite.comp.spv", rayTraceFinalWritePipeline);
        makeComputePipeline("build/shaders/FSR.comp.spv",                               FSRPipeline);
        makeComputePipeline("build/shaders/FSRSharpen.comp.spv",                 FSRSharpenPipeline);
    }
}


void ShowcaseApp::createRenderPass()
{
    VkAttachmentDescription color{};
    color.format = swapChainImageFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = 1;
    createInfo.pAttachments = &color;
    createInfo.subpassCount = 1;
    createInfo.pSubpasses = &subpass;
    createInfo.dependencyCount = 1;
    createInfo.pDependencies = &dep;

    if (vkCreateRenderPass(device, &createInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create render pass!");
}

void ShowcaseApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        VkImageView attachments[1] = { swapChainImageViews[i] };

        VkFramebufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        createInfo.renderPass = renderPass;
        createInfo.attachmentCount = 1;
        createInfo.pAttachments = attachments;
        createInfo.width  = swapChainExtent.width;
        createInfo.height = swapChainExtent.height;
        createInfo.layers = 1;

        if (vkCreateFramebuffer(device, &createInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create framebuffer!");
    }
}

void ShowcaseApp::createGraphicsDescriptors()
{
    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    if (vkCreateSampler(device, &samplerCreateInfo, nullptr, &offscreenSampler) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create sampler!");

    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = 1;
    layoutCreateInfo.pBindings = &b0;

    if (vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &graphicsSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics descriptor set layout!");

    VkDescriptorPoolSize p{};
    p.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    p.descriptorCount = SwapChain::MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.maxSets = SwapChain::MAX_FRAMES_IN_FLIGHT;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &p;

    if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &graphicsDescPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics descriptor pool!");

    std::array<VkDescriptorSetLayout, SwapChain::MAX_FRAMES_IN_FLIGHT> layouts{};
    layouts.fill(graphicsSetLayout);
    
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = graphicsDescPool;
    allocateInfo.descriptorSetCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    allocateInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(device, &allocateInfo, graphicsSets) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate graphics descriptor set!");
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        char name[64];
        std::snprintf(name, sizeof(name), "GraphicsSet[%u]", i);
        setDescriptorSetName(graphicsSets[i], name);
    }

    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {

        VkDescriptorImageInfo img{};
        img.sampler = offscreenSampler;
        img.imageView = (SimpleRayTrace ? offscreenView[i] : fsrView[i]);
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = graphicsSets[i];
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &img;
    
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
}

void ShowcaseApp::createFullscreenGraphicsPipeline()
{
    VkPipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = 1;
    layoutCreateInfo.pSetLayouts = &graphicsSetLayout;

    if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics pipeline layout!");

    auto vertCode = Pipeline::readFile("build/shaders/fullScreen.vert.spv");
    auto fragCode = Pipeline::readFile("build/shaders/fullScreen.frag.spv");

    VkShaderModule vert = createShaderModule(vertCode);
    VkShaderModule frag = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
    vertShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageCreateInfo.module = vert;
    vertShaderStageCreateInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
    fragShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageCreateInfo.module = frag;
    fragShaderStageCreateInfo.pName = "main";

    VkPipelineShaderStageCreateInfo stages[2] = { vertShaderStageCreateInfo, fragShaderStageCreateInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
    vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo{};
    inputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width  = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportCreateInfo{};
    viewportCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportCreateInfo.viewportCount = 1;
    viewportCreateInfo.pViewports = &viewport;
    viewportCreateInfo.scissorCount = 1;
    viewportCreateInfo.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizationCreateInfo{};
    rasterizationCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationCreateInfo.cullMode = VK_CULL_MODE_NONE;
    rasterizationCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationCreateInfo.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampleCreateInfo{};
    multisampleCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlendCreateInfo{};
    colorBlendCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendCreateInfo.attachmentCount = 1;
    colorBlendCreateInfo.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = stages;
    pipelineCreateInfo.pVertexInputState   = &vertexInputCreateInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
    pipelineCreateInfo.pViewportState      = &viewportCreateInfo;
    pipelineCreateInfo.pRasterizationState = &rasterizationCreateInfo;
    pipelineCreateInfo.pMultisampleState   = &multisampleCreateInfo;
    pipelineCreateInfo.pDepthStencilState  = nullptr;
    pipelineCreateInfo.pColorBlendState    = &colorBlendCreateInfo;
    pipelineCreateInfo.layout = graphicsPipelineLayout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics pipeline!");

    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
}


// ~~~~~ Destruction ~~~~~~

void ShowcaseApp::destroyNRD()
{
    // 1) Destroy NRD Integration (this cleans all NRI resources allocated inside)
    nrdIntegration.Destroy();

    // 2) Destroy NRD OUT_* per-frame images (diff + spec)
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        auto& frame = nrdFrameImages[i];

        if (frame.diffView)   { vkDestroyImageView(device, frame.diffView,  nullptr); frame.diffView  = VK_NULL_HANDLE; }
        if (frame.diffImage)  { vkDestroyImage(device,     frame.diffImage, nullptr); frame.diffImage = VK_NULL_HANDLE; }
        if (frame.diffMemory) { vkFreeMemory(device,       frame.diffMemory,nullptr); frame.diffMemory= VK_NULL_HANDLE; }

        if (frame.specView)   { vkDestroyImageView(device, frame.specView,  nullptr); frame.specView  = VK_NULL_HANDLE; }
        if (frame.specImage)  { vkDestroyImage(device,     frame.specImage, nullptr); frame.specImage = VK_NULL_HANDLE; }
        if (frame.specMemory) { vkFreeMemory(device,       frame.specMemory,nullptr); frame.specMemory= VK_NULL_HANDLE; }

        frame.valid = false;
    }

    // 3) Destroy NRD IN_* per-frame images, if you have them as NrdTexture arrays
    auto destroyNrdTexture = [&](NrdTexture& t)
    {
        if (t.view)   { vkDestroyImageView(device, t.view,  nullptr); t.view  = VK_NULL_HANDLE; }
        if (t.image)  { vkDestroyImage(device,     t.image, nullptr); t.image = VK_NULL_HANDLE; }
        if (t.memory) { vkFreeMemory(device,       t.memory,nullptr); t.memory= VK_NULL_HANDLE; }
        t.width  = 0;
        t.height = 0;
        t.format = VK_FORMAT_UNDEFINED;
    };

    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        destroyNrdTexture(nrdInputs[i].diffRadianceHit);
        destroyNrdTexture(nrdInputs[i].specRadianceHit);
        destroyNrdTexture(nrdInputs[i].normalRoughness);
        destroyNrdTexture(nrdInputs[i].viewZ);
        destroyNrdTexture(nrdInputs[i].motionVec);
        // plus any extra IN_* you add later
    }

    // 4) Reset frame counters / prev sizes if you want
    nrdFrameIndex          = 0;
    nrdResourceSizePrev[0] = 0;
    nrdResourceSizePrev[1] = 0;
    nrdRectSizePrev[0]     = 0;
    nrdRectSizePrev[1]     = 0;
}


void ShowcaseApp::destroyWavefrontBuffers()
{
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        auto destroy = [&](VkBuffer& buf, VkDeviceMemory& mem)
        {
            if (buf) { vkDestroyBuffer(device, buf, nullptr); buf = VK_NULL_HANDLE; }
            if (mem) { vkFreeMemory(device, mem, nullptr);     mem = VK_NULL_HANDLE; }
        };

        destroy(pathHeaderBuf[i],   pathHeaderMem[i]);
        destroy(rayBuf[i],          rayMem[i]);
        destroy(hitIdsBuf[i],       hitIdsMem[i]);
        destroy(hitDataBuf[i],      hitDataMem[i]);
        destroy(radianceBuf[i],     radianceMem[i]);
        destroy(bsdfSampleBuf[i],   bsdfSampleMem[i]);
        destroy(lightSampleBuf[i],  lightSampleMem[i]);
        destroy(shadowRayBuf[i],    shadowRayMem[i]);
        destroy(shadowResultBuf[i], shadowResultMem[i]);
        destroy(pathQueueBuf[i],    pathQueueMem[i]);
        destroy(pixelStatsBuf[i],      pixelStatsMem[i]);
        destroy(highVarPixelBuf[i],    highVarPixelMem[i]);
        destroy(adaptiveCountersBuf[i], adaptiveCountersMem[i]);
    }
}

void ShowcaseApp::destroySceneTextures()
{
    if (device == VK_NULL_HANDLE)
        return;
    for (auto& texture : gpuTextures)
        destroyGpuTexture(device, texture);
    gpuTextures.clear();

    for (auto& t : nrdPermanentTextures)
        destroyNRDTexture(device, t);
    for (auto& t : nrdTransientTextures)
        destroyNRDTexture(device, t);
    nrdPermanentTextures.clear();
    nrdTransientTextures.clear();
}

void ShowcaseApp::destroyGraphicsDescriptors()
{
    if (graphicsDescPool)
    {
        vkDestroyDescriptorPool(device, graphicsDescPool, nullptr);
        graphicsDescPool = VK_NULL_HANDLE;
    }
    if (graphicsSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, graphicsSetLayout, nullptr);
        graphicsSetLayout = VK_NULL_HANDLE;
    }
    if (offscreenSampler)
    {
        vkDestroySampler(device, offscreenSampler, nullptr);
        offscreenSampler = VK_NULL_HANDLE;
    }
}

void ShowcaseApp::destroyFramebuffers()
{
    for (auto fb : swapChainFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    swapChainFramebuffers.clear();
}

void ShowcaseApp::destroyComputeDescriptors()
{
    if (computeDescPool)
    {
        vkDestroyDescriptorPool(device, computeDescPool, nullptr);
        computeDescPool = VK_NULL_HANDLE;
    }

    if (computeStaticSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, computeStaticSetLayout, nullptr);
        computeStaticSetLayout = VK_NULL_HANDLE;
    }

    if (computeFrameSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, computeFrameSetLayout, nullptr);
        computeFrameSetLayout = VK_NULL_HANDLE;
    }

    if (computeDynamicSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, computeDynamicSetLayout, nullptr);
        computeDynamicSetLayout = VK_NULL_HANDLE;
    }

    computeStaticSet = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        computeFrameSets[i]   = VK_NULL_HANDLE;
        computeDynamicSets[i] = VK_NULL_HANDLE;
    }
}


void ShowcaseApp::destroyOffscreenTarget()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (!offscreenValid[i])
        {
            if (computeDone[i])   { vkDestroySemaphore(device, computeDone[i], nullptr); computeDone[i] = VK_NULL_HANDLE; }
            if (offscreenView[i]) { vkDestroyImageView(device, offscreenView[i], nullptr); offscreenView[i] = VK_NULL_HANDLE; }
            if (offscreenImage[i]){ vkDestroyImage(device, offscreenImage[i], nullptr);    offscreenImage[i] = VK_NULL_HANDLE; }
            if (offscreenMemory[i]){vkFreeMemory(device, offscreenMemory[i], nullptr);     offscreenMemory[i] = VK_NULL_HANDLE; }
            continue;
        }

        if (computeDone[i])   { vkDestroySemaphore(device, computeDone[i], nullptr); computeDone[i] = VK_NULL_HANDLE; }
        if (offscreenView[i]) { vkDestroyImageView(device, offscreenView[i], nullptr); offscreenView[i] = VK_NULL_HANDLE; }
        if (offscreenImage[i]){ vkDestroyImage(device, offscreenImage[i], nullptr);    offscreenImage[i] = VK_NULL_HANDLE; }
        if (offscreenMemory[i]){vkFreeMemory(device, offscreenMemory[i], nullptr);     offscreenMemory[i] = VK_NULL_HANDLE; }

        offscreenValid[i] = false;
        
    }
}

void ShowcaseApp::destroyFSRTarget()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (!fsrValid[i])
        {
            if (fsrView[i]) { vkDestroyImageView(device, fsrView[i], nullptr); fsrView[i] = VK_NULL_HANDLE; }
            if (fsrImage[i]){ vkDestroyImage(device, fsrImage[i], nullptr);    fsrImage[i] = VK_NULL_HANDLE; }
            if (fsrMemory[i]){vkFreeMemory(device, fsrMemory[i], nullptr);     fsrMemory[i] = VK_NULL_HANDLE; }
            continue;
        }

        if (fsrView[i]) { vkDestroyImageView(device, fsrView[i], nullptr); fsrView[i] = VK_NULL_HANDLE; }
        if (fsrImage[i]){ vkDestroyImage(device, fsrImage[i], nullptr);    fsrImage[i] = VK_NULL_HANDLE; }
        if (fsrMemory[i]){vkFreeMemory(device, fsrMemory[i], nullptr);     fsrMemory[i] = VK_NULL_HANDLE; }

        fsrValid[i] = false;
        
    }
}

void ShowcaseApp::destroyNrdTargets()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        auto &frame = nrdFrameImages[i];

        if (frame.diffView)
        {
            vkDestroyImageView(device, frame.diffView, nullptr);
            frame.diffView = VK_NULL_HANDLE;
        }
        if (frame.diffImage)
        {
            vkDestroyImage(device, frame.diffImage, nullptr);
            frame.diffImage = VK_NULL_HANDLE;
        }
        if (frame.diffMemory)
        {
            vkFreeMemory(device, frame.diffMemory, nullptr);
            frame.diffMemory = VK_NULL_HANDLE;
        }
        if (frame.specView)
        {
            vkDestroyImageView(device, frame.specView, nullptr);
            frame.specView = VK_NULL_HANDLE;
        }
        if (frame.specImage)
        {
            vkDestroyImage(device, frame.specImage, nullptr);
            frame.specImage = VK_NULL_HANDLE;
        }
        if (frame.specMemory)
        {
            vkFreeMemory(device, frame.specMemory, nullptr);
            frame.specMemory = VK_NULL_HANDLE;
        }

        frame.valid = false;
    }
}


void ShowcaseApp::destroySSBOdata()
{
    if (sbvhNodesBuffer)
        vkDestroyBuffer(device, sbvhNodesBuffer, nullptr);
    if (sbvhNodesMemory)
        vkFreeMemory(device, sbvhNodesMemory, nullptr);

    if (triangleBuffer)
        vkDestroyBuffer(device, triangleBuffer, nullptr);
    if (triangleMemory)
        vkFreeMemory(device, triangleMemory, nullptr);

    if (shadingBuffer)
        vkDestroyBuffer(device, shadingBuffer, nullptr);
    if (shadingMemory)
        vkFreeMemory(device, shadingMemory, nullptr);

    if (materialBuffer)
        vkDestroyBuffer(device, materialBuffer, nullptr);
    if (materialMemory)
        vkFreeMemory(device, materialMemory, nullptr);
    
    if (emissiveTriangleBuffer)
        vkDestroyBuffer(device, emissiveTriangleBuffer, nullptr);
    if (emissiveTriangleMemory)
        vkFreeMemory(device, emissiveTriangleMemory, nullptr);

    if (lightProbBuffer)
        vkDestroyBuffer(device, lightProbBuffer, nullptr);
    if (lightProbMemory)
        vkFreeMemory(device, lightProbMemory, nullptr);

    if (lightPdfBuffer)
        vkDestroyBuffer(device, lightPdfBuffer, nullptr);
    if (lightPdfMemory)
        vkFreeMemory(device, lightPdfMemory, nullptr);

    if (lightAliasBuffer)
        vkDestroyBuffer(device, lightAliasBuffer, nullptr);
    if (lightAliasMemory)
        vkFreeMemory(device, lightAliasMemory, nullptr);

    if (triToLightIdxBuffer)
        vkDestroyBuffer(device, triToLightIdxBuffer, nullptr);
    if (triToLightIdxMemory)
        vkFreeMemory(device, triToLightIdxMemory, nullptr);

}
