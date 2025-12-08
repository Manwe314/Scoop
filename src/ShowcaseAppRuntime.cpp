#include "ShowcaseApp.hpp"
#include <iostream>
#define A_CPU 1
#include "ffx_a.h"
#define FSR_EASU_F 1
#include "ffx_fsr1.h"
typedef uint32_t AU4[4];
#include <cstdio>

// --------- helpers -----------

constexpr uint32_t FLAG_B            = 1u << 31;
constexpr uint32_t FLAG_INSTANCE_SEL = 1u << 30;
constexpr uint32_t B_INTERP_SHIFT    = 20;
constexpr uint32_t B_INTERP_BITS     = 10;
constexpr uint32_t B_INTERP_MAX      = (1u << B_INTERP_BITS) - 1u;
constexpr uint32_t B_INTERP_MASK     = B_INTERP_MAX << B_INTERP_SHIFT;
constexpr uint32_t MASK_INSTANCE_IDX = 0xFu; 
constexpr uint32_t FLAG_DEBUG_NRD_VALIDATION = 1u << 19;
constexpr uint32_t FLAG_DEBUG_NRD_INPUTS     = 1u << 18;

static inline uint32_t qpBase(uint32_t fi) { return fi * 4u; }
static inline uint32_t qpGfxBegin(uint32_t fi){ return qpBase(fi) + 0; }
static inline uint32_t qpGfxEnd  (uint32_t fi){ return qpBase(fi) + 1; }
static inline uint32_t qpCmpBegin(uint32_t fi){ return qpBase(fi) + 2; }
static inline uint32_t qpCmpEnd  (uint32_t fi){ return qpBase(fi) + 3; }


static VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); }

static inline VkDeviceSize nonZero(VkDeviceSize v, VkDeviceSize min = 16)
{
    return v ? v : min;
}

void ShowcaseApp::setObjectName(VkObjectType type, uint64_t handle, const char* name)
{
    if (device == VK_NULL_HANDLE || handle == 0 || name == nullptr)
        return;
    auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    if (!func)
        return;

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = name;
    func(device, &info);
}

static float halton(uint32_t index, uint32_t base)
{
    float f = 1.0f;
    float r = 0.0f;
    uint32_t i = index;

    while (i > 0)
    {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }

    return r;
}

static glm::vec2 getHaltonJitterPx(uint32_t index)
{
    float jx = halton(index + 1, 2) - 0.5f;
    float jy = halton(index + 1, 3) - 0.5f;
    return glm::vec2(jx, jy);
}

void ShowcaseApp::setImageName(VkImage image, const char* name)
{
    setObjectName(VK_OBJECT_TYPE_IMAGE,
                  reinterpret_cast<uint64_t>(image),
                  name);
}

void ShowcaseApp::setImageViewName(VkImageView view, const char* name)
{
    setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW,
                  reinterpret_cast<uint64_t>(view),
                  name);
}

void ShowcaseApp::setDescriptorSetName(VkDescriptorSet set, const char* name)
{
    setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET,
                  reinterpret_cast<uint64_t>(set),
                  name);
}

inline AABB boundsOfRange(const std::vector<InstanceData>& inst, const std::vector<uint32_t>& idx, uint32_t first, uint32_t count)
{
    AABB b = makeEmptyAABB();
    for (uint32_t i = 0; i < count; ++i)
        mergeInto(b, inst[idx[first + i]].worldAABB);
    return b;
}

inline uint32_t makeNode(TLAS& tlas, const TLASNode& n)
{
    tlas.nodes.push_back(n);
    return static_cast<uint32_t>(tlas.nodes.size() - 1);
}

inline uint32_t buildRecursive(TLAS& tlas, std::vector<uint32_t>& idx, uint32_t first, uint32_t count)
{
    TLASNode node{};
    node.bounds = boundsOfRange(tlas.instances, idx, first, count);

    const uint32_t LEAF_THRESHOLD = 1;
    if (count <= LEAF_THRESHOLD)
    {
        node.first = first;
        node.count = count;
        return makeNode(tlas, node);
    }

    AABB cb;
    for (uint32_t i = 0; i < count; ++i)
        cb = merge(cb, AABB{ centroid(tlas.instances[idx[first + i]].worldAABB), centroid(tlas.instances[idx[first + i]].worldAABB) });
    
    int axis = widestAxis(cb);

    if (cb.min[axis] == cb.max[axis])
    {
        uint32_t mid = first + count / 2;
        uint32_t left  = buildRecursive(tlas, idx, first, mid - first);
        uint32_t right = buildRecursive(tlas, idx, mid,   first + count - mid);
        node.left = int32_t(left);
        node.right = int32_t(right);
        return makeNode(tlas, node);
    }

    uint32_t mid = first + count / 2;
    std::nth_element(idx.begin() + first, idx.begin() + mid, idx.begin() + first + count, [&](uint32_t a, uint32_t b)
    {
        return centroid(tlas.instances[a].worldAABB)[axis] <
               centroid(tlas.instances[b].worldAABB)[axis];
    });

    uint32_t left  = buildRecursive(tlas, idx, first, mid - first);
    uint32_t right = buildRecursive(tlas, idx, mid,   first + count - mid);

    node.left  = static_cast<int32_t>(left);
    node.right = static_cast<int32_t>(right);
    return makeNode(tlas, node);
}

inline void updateInstances(std::vector<InstanceData>& instances, Scene& scene)
{
    for (int i = 0; i < scene.objects.size(); i++)
    {
        auto& object = scene.objects[i];
        float speed = object.animation.SpeedScalar * 0.02;
        object.transform.rotation = glm::mod(object.transform.rotation + object.animation.rotationSpeedPerAxis * speed, 360.0f);
        instances[i].worldAABB = object.transformAABB();
        instances[i].modelToWorld = object.transform.affineTransform();
        instances[i].worldToModel = affineInverse(instances[i].modelToWorld);
    }
}

inline TLAS buildTLAS(const std::vector<InstanceData>& instances)
{
    TLAS tlas;
    tlas.instances = instances;
    tlas.root = 0;

    const uint32_t N = static_cast<uint32_t>(instances.size());
    tlas.instanceIndices.resize(N);
    for (uint32_t i = 0; i < N; ++i)
        tlas.instanceIndices[i] = i;

    if (N == 0)
        return tlas;

    tlas.root = buildRecursive(tlas, tlas.instanceIndices, 0, N);

    return tlas;
}

inline TLASNodeGPU packNode(const TLASNode& node)
{
    TLASNodeGPU out{};
    out.bmin = glm::vec4(node.bounds.min, 0.0f);
    out.bmax = glm::vec4(node.bounds.max, 0.0f);
    uint32_t U = 0xFFFFFFFFu;
    out.meta = glm::uvec4(node.first, node.count,
                node.left  >= 0 ? uint32_t(node.left)  : U,
                node.right >= 0 ? uint32_t(node.right) : U);
    return out;
}

inline InstanceDataGPU packInstance(const InstanceData& data, AABB modelBB)
{
    InstanceDataGPU out{};

    const Mat4 M = affineToMat4(data.modelToWorld);
    const Mat4 W = affineToMat4(data.worldToModel);

    out.modelToWorld[0] = glm::vec4(M.x[0], M.y[0], M.z[0], M.w[0]);
    out.modelToWorld[1] = glm::vec4(M.x[1], M.y[1], M.z[1], M.w[1]);
    out.modelToWorld[2] = glm::vec4(M.x[2], M.y[2], M.z[2], M.w[2]);

    out.worldToModel[0] = glm::vec4(W.x[0], W.y[0], W.z[0], W.w[0]);
    out.worldToModel[1] = glm::vec4(W.x[1], W.y[1], W.z[1], W.w[1]);
    out.worldToModel[2] = glm::vec4(W.x[2], W.y[2], W.z[2], W.w[2]);

    out.aabbMin = glm::vec4(modelBB.min, 0.0f);
    out.aabbMax = glm::vec4(modelBB.max, 0.0f);

    out.bases0 = glm::uvec4(data.nodeBase, data.triBase, data.shadeTriBase, data.materialBase);
    out.bases1 = glm::uvec4(data.textureBase, 0u, 0u, data.lightMapBase);
    return out;
}


inline Mat4 LookAtRH(const glm::vec3& eye,
                     const glm::vec3& center,
                     const glm::vec3& up)
{
    const glm::vec3 f = glm::normalize(center - eye);
    const glm::vec3 s = glm::normalize(glm::cross(f, up));
    const glm::vec3 u = glm::cross(s, f);

    Mat4 Result(1.0f);

    Result[0][0] = s.x;
    Result[1][0] = s.y;
    Result[2][0] = s.z;
    Result[3][0] = -glm::dot(s, eye);

    Result[0][1] = u.x;
    Result[1][1] = u.y;
    Result[2][1] = u.z;
    Result[3][1] = -glm::dot(u, eye);

    Result[0][2] = -f.x;
    Result[1][2] = -f.y;
    Result[2][2] = -f.z;
    Result[3][2] =  glm::dot(f, eye);

    return Result;
}


inline Mat4 PerspectiveRH_ZO(float fovY, float aspect, float zNear, float zFar)
{
    const float tanHalfFovy = std::tan(fovY * 0.5f);

    Mat4 Result(0.0f);

    Result[0][0] = 1.0f / (aspect * tanHalfFovy);
    Result[1][1] = 1.0f / (tanHalfFovy);
    Result[2][2] = zFar / (zNear - zFar);
    Result[2][3] = -1.0f;
    Result[3][2] = -(zFar * zNear) / (zFar - zNear);
    return Result;
}



inline ParamsGPU makeParamsForVulkan(
    VkExtent2D extent,
    uint32_t   rootIndex,
    float      time,
    glm::vec3  camPos,
    glm::vec3  camTarget = glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3  camUp     = glm::vec3(0.0f, 1.0f, 0.0f),
    float      fovY_deg  = 60.0f,
    float      zNear     = 0.1f,
    float      zFar      = 2000.0f,
    uint32_t   flags     = 0,
    uint32_t   frameIndex = 1,
    Mat4* outViewProj    = nullptr,
    Mat4* outViewProjInv = nullptr,
    Mat4* outView        = nullptr,
    Mat4* outProj        = nullptr
)
{
    const float aspect = float(extent.width) / float(std::max(1u, extent.height));

    Mat4 view = LookAtRH(camPos, camTarget, camUp);
    Mat4 proj = PerspectiveRH_ZO(glm::radians(fovY_deg), aspect, zNear, zFar);

    // Vulkan-style projection flip
    proj[1][1] *= -1.0f;

    Mat4 viewProj = proj * view;
    Mat4 viewProjInv;
    inverse(viewProj, viewProjInv);

    if (outViewProj)     *outViewProj    = viewProj;
    if (outViewProjInv)  *outViewProjInv = viewProjInv;
    if (outView)         *outView        = view;   // NEW
    if (outProj)         *outProj        = proj;   // NEW

    ParamsGPU p{};
    p.viewProjInv = viewProjInv;
    p.camPos_time = glm::vec4(camPos, float(time * frameIndex));
    p.imageSize   = glm::uvec2(extent.width, extent.height);
    p.rootIndex   = rootIndex;
    p.flags       = flags;
    return p;
}


void ShowcaseApp::ensureBufferCapacity(
    VkBuffer& buf, VkDeviceMemory& mem,
    VkDeviceSize neededSize,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags)
{
    
    neededSize = nonZero(neededSize);

    if (buf != VK_NULL_HANDLE)
    {
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, buf, &req);
        if (req.size >= neededSize)
            return;
        vkDestroyBuffer(device, buf, nullptr);
        vkFreeMemory(device, mem, nullptr);
        buf = VK_NULL_HANDLE;
        mem = VK_NULL_HANDLE;
    }

    createBuffer(device, physicalDevice, neededSize, usage, flags, buf, mem);
}

void ShowcaseApp::uploadTLASForFrame(uint32_t frameIndex,
                                     const std::vector<TLASNodeGPU>& tlasNodes,
                                     const std::vector<InstanceDataGPU>& tlasInstances,
                                     const std::vector<uint32_t>& instanceIndices)
{
    if (computeFences[frameIndex] != VK_NULL_HANDLE)
        vkWaitForFences(device, 1, &computeFences[frameIndex], VK_TRUE, UINT64_MAX);
    auto& frame = frameUpload[frameIndex];

    const VkDeviceSize nodesBytes = VkDeviceSize(tlasNodes.size()) * sizeof(TLASNodeGPU);
    const VkDeviceSize instBytes  = VkDeviceSize(tlasInstances.size()) * sizeof(InstanceDataGPU);
    const VkDeviceSize idxBytes   = VkDeviceSize(instanceIndices.size()) * sizeof(uint32_t);
    const auto& prevInstSrc = hasPrevInstanceData ? prevInstancesGPU : tlasInstances;
    const auto& prevIdxSrc  = hasPrevInstanceData ? prevInstanceIndices : instanceIndices;
    const VkDeviceSize prevInstBytes = VkDeviceSize(prevInstSrc.size()) * sizeof(InstanceDataGPU);
    const VkDeviceSize prevIdxBytes  = VkDeviceSize(prevIdxSrc.size()) * sizeof(uint32_t);

    ensureBufferCapacity(tlasNodesBuf[frameIndex], tlasNodesMem[frameIndex], nodesBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ensureBufferCapacity(tlasInstBuf[frameIndex],  tlasInstMem[frameIndex],  instBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ensureBufferCapacity(prevTlasInstBuf[frameIndex], prevTlasInstMem[frameIndex],  prevInstBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ensureBufferCapacity(tlasIdxBuf[frameIndex],   tlasIdxMem[frameIndex],   idxBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ensureBufferCapacity(prevTlasIdxBuf[frameIndex],   prevTlasIdxMem[frameIndex],   prevIdxBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkDeviceSize off = 0;
    const VkDeviceSize A = 16;

    const VkDeviceSize nodesOff = off;
    off = alignUp(off + nodesBytes, A);
    const VkDeviceSize instOff  = off;
    off = alignUp(off + instBytes,  A);
    const VkDeviceSize idxOff   = off;
    off = alignUp(off + idxBytes,   A);
    const VkDeviceSize prevInstOff = off;
    off = alignUp(off + prevInstBytes, A);
    const VkDeviceSize prevIdxOff  = off;
    off = alignUp(off + prevIdxBytes,  A);
    const VkDeviceSize total    = off;

    ensureUploadStagingCapacity(frameIndex, total);

    if (nodesBytes)
        std::memcpy(static_cast<char*>(frame.mapped) + nodesOff, tlasNodes.data(), size_t(nodesBytes));
    if (instBytes) 
        std::memcpy(static_cast<char*>(frame.mapped) + instOff,  tlasInstances.data(), size_t(instBytes));
    if (idxBytes)
        std::memcpy(static_cast<char*>(frame.mapped) + idxOff,   instanceIndices.data(), size_t(idxBytes));
    if (prevInstBytes)
        std::memcpy(static_cast<char*>(frame.mapped) + prevInstOff, prevInstSrc.data(), size_t(prevInstBytes));
    if (prevIdxBytes)
        std::memcpy(static_cast<char*>(frame.mapped) + prevIdxOff,  prevIdxSrc.data(), size_t(prevIdxBytes));
    

    auto copyIf = [&](VkBuffer dst, VkDeviceSize size, VkDeviceSize srcOff)
    {
        if (!size)
            return;
        VkBufferCopy c{ srcOff, 0, size };
        vkCmdCopyBuffer(frame.uploadCB, frame.staging, dst, 1, &c);
    };

    copyIf(tlasNodesBuf[frameIndex], nodesBytes, nodesOff);
    copyIf(tlasInstBuf[frameIndex],  instBytes,  instOff);
    copyIf(tlasIdxBuf[frameIndex],   idxBytes,   idxOff);
    copyIf(prevTlasInstBuf[frameIndex], prevInstBytes, prevInstOff);
    copyIf(prevTlasIdxBuf[frameIndex],  prevIdxBytes,  prevIdxOff);

    if constexpr (!SimpleRayTrace)
    {
        if (!prevViewZInitialized[frameIndex])
        {
            imageBarrier(frame.uploadCB, prevViewZImage[frameIndex],
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            VkClearColorValue clear{};
            clear.float32[0] = 1e30f;
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;
            vkCmdClearColorImage(frame.uploadCB, prevViewZImage[frameIndex], VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
            prevViewZInitialized[frameIndex] = true;
        }
    }

    if (hasDedicatedTransfer && computeFamily != transferFamily)
    {
        std::vector<VkBufferMemoryBarrier> rels;
        auto addRel = [&](VkBuffer b, VkDeviceSize sz)
        {
            if (!b || !sz)
                return;
            VkBufferMemoryBarrier bb{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
            bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bb.dstAccessMask = 0;
            bb.srcQueueFamilyIndex = transferFamily;
            bb.dstQueueFamilyIndex = computeFamily;
            bb.buffer = b;
            bb.offset = 0;
            bb.size   = VK_WHOLE_SIZE;
            rels.push_back(bb);
        };
        addRel(tlasNodesBuf[frameIndex], nodesBytes);
        addRel(tlasInstBuf[frameIndex],  instBytes);
        addRel(tlasIdxBuf[frameIndex],   idxBytes);
        addRel(prevTlasInstBuf[frameIndex], prevInstBytes);
        addRel(prevTlasIdxBuf[frameIndex],  prevIdxBytes);

        if (!rels.empty())
        {
            vkCmdPipelineBarrier(frame.uploadCB,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, (uint32_t)rels.size(), rels.data(), 0, nullptr);
        }
    }
    else
    {
        std::vector<VkBufferMemoryBarrier> vis;
        auto addVis = [&](VkBuffer b, VkDeviceSize sz)
        {
            if (!b || !sz)
                return;
            VkBufferMemoryBarrier bb{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
            bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.buffer = b;
            bb.offset = 0;
            bb.size   = VK_WHOLE_SIZE;
            vis.push_back(bb);
        };
        addVis(tlasNodesBuf[frameIndex], nodesBytes);
        addVis(tlasInstBuf[frameIndex],  instBytes);
        addVis(tlasIdxBuf[frameIndex],   idxBytes);
        addVis(prevTlasInstBuf[frameIndex], prevInstBytes);
        addVis(prevTlasIdxBuf[frameIndex],  prevIdxBytes);

        if (!vis.empty())
        {
            vkCmdPipelineBarrier(frame.uploadCB,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(vis.size()), vis.data(), 0, nullptr);
        }
    }

    VkDescriptorBufferInfo b0{ tlasNodesBuf[frameIndex], 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b1{ tlasInstBuf[frameIndex],  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b2{ tlasIdxBuf[frameIndex],   0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b3{ prevTlasInstBuf[frameIndex], 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b4{ prevTlasIdxBuf[frameIndex],  0, VK_WHOLE_SIZE };

    std::array<VkWriteDescriptorSet, 5> w{};
    auto initWrite = [&](VkWriteDescriptorSet& write, uint32_t binding, VkDescriptorBufferInfo* info)
    {
        write = {};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.dstSet          = computeFrameSets[frameIndex];
        write.dstBinding      = binding;
        write.pBufferInfo     = info;
    };

    // set 1, binding 0: TLAS nodes
    initWrite(w[0], 0, &b0);

    // set 1, binding 1: TLAS instances
    initWrite(w[1], 1, &b1);

    // set 1, binding 2: TLAS indices
    initWrite(w[2], 2, &b2);

    uint32_t writeCount = 3;

    if constexpr (!SimpleRayTrace)
    {
        // set 1, binding 19: prev TLAS instances
        initWrite(w[3], 19, &b3);

        // set 1, binding 20: prev TLAS indices
        initWrite(w[4], 20, &b4);

        writeCount = 5;
    }

    vkUpdateDescriptorSets(device, writeCount, w.data(), 0, nullptr);

    prevInstancesGPU    = tlasInstances;
    prevInstanceIndices = instanceIndices;
    hasPrevInstanceData = true;
}


void ShowcaseApp::createCommandPoolAndBuffers()
{
    QueueFamiliyIndies indices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo graphicsCreateInfo{};
    graphicsCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    graphicsCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
    graphicsCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &graphicsCreateInfo, nullptr, &graphicsCommandPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create command pool!");


    VkCommandPoolCreateInfo computeCreateInfo{};
    computeCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    computeCreateInfo.queueFamilyIndex = indices.computeFamily.value();
    computeCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &computeCreateInfo, nullptr, &computeCommandPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create command pool!");

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = static_cast<uint32_t>(SwapChain::MAX_FRAMES_IN_FLIGHT);
    allocateInfo.commandPool = graphicsCommandPool;

    if (vkAllocateCommandBuffers(device, &allocateInfo, graphicsCommandBuffers) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate command buffers!");
    allocateInfo.commandPool = computeCommandPool;
    if (vkAllocateCommandBuffers(device, &allocateInfo, computeCommandBuffers) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate command buffers!");

}

void ShowcaseApp::imageBarrier(VkCommandBuffer cmd, VkImage image,
                            VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                            VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
                            VkImageLayout oldLayout, VkImageLayout newLayout,
                            uint32_t srcQue, uint32_t dstQue)
{
    VkImageMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = srcAccess;
    memoryBarrier.dstAccessMask = dstAccess;
    memoryBarrier.oldLayout = oldLayout;
    memoryBarrier.newLayout = newLayout;
    memoryBarrier.srcQueueFamilyIndex = srcQue;
    memoryBarrier.dstQueueFamilyIndex = dstQue;
    memoryBarrier.image = image;
    memoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    memoryBarrier.subresourceRange.baseMipLevel = 0;
    memoryBarrier.subresourceRange.levelCount = 1;
    memoryBarrier.subresourceRange.baseArrayLayer = 0;
    memoryBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &memoryBarrier);
}

void ShowcaseApp::transitionNrdInputsForDenoising(VkCommandBuffer cmd, uint32_t frameIndex)
{
    auto transitionInput = [&](VkImage img)
    {
        if (img == VK_NULL_HANDLE)
            return;

        imageBarrier(cmd, img,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    };

    transitionInput(nrdInputs[frameIndex].diffRadianceHit.image);
    transitionInput(nrdInputs[frameIndex].specRadianceHit.image);
    transitionInput(nrdInputs[frameIndex].normalRoughness.image);
    transitionInput(nrdInputs[frameIndex].viewZ.image);
    transitionInput(nrdInputs[frameIndex].motionVec.image);

    nrdInputsSampled[frameIndex] = true;
}

void ShowcaseApp::recordComputeCommands(uint32_t i)
{
    VkCommandBuffer cmd = computeCommandBuffers[i];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);
    
    vkCmdResetQueryPool(cmd, queryPool, qpCmpBegin(i), 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, qpCmpBegin(i));
    
    if (!offscreenInitialized[i])
    {
        imageBarrier(cmd, offscreenImage[i],
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
        offscreenInitialized[i] = true;
    }
    else
    {
        imageBarrier(cmd, offscreenImage[i],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    }
    
    if constexpr (!SimpleRayTrace)
    {
        if (!fsrInitialized[i])
        {
            imageBarrier(cmd, fsrImage[i],
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            fsrInitialized[i] = true;
        }
        else
        {
            imageBarrier(cmd, fsrImage[i],
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                         VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
        }

        if (!nrdInputsInitialized[i])
        {
            auto initNrdInImage = [&](VkImage img)
            {
                if (img == VK_NULL_HANDLE)
                    return;

                imageBarrier(cmd, img,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            };

            initNrdInImage(nrdInputs[i].diffRadianceHit.image);
            initNrdInImage(nrdInputs[i].specRadianceHit.image);
            initNrdInImage(nrdInputs[i].normalRoughness.image);
            initNrdInImage(nrdInputs[i].viewZ.image);
            initNrdInImage(nrdInputs[i].motionVec.image);

            nrdInputsInitialized[i] = true;
        }
        else if (nrdInputsSampled[i])
        {
            auto makeWritable = [&](VkImage img)
            {
                if (img == VK_NULL_HANDLE)
                    return;

                imageBarrier(cmd, img,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            };

            makeWritable(nrdInputs[i].diffRadianceHit.image);
            makeWritable(nrdInputs[i].specRadianceHit.image);
            makeWritable(nrdInputs[i].normalRoughness.image);
            makeWritable(nrdInputs[i].viewZ.image);
            makeWritable(nrdInputs[i].motionVec.image);

            nrdInputsSampled[i] = false;
        }

        if (!nrdOutputsInitialized[i])
        {
            auto initNrdOutImage = [&](VkImage img)
            {
                if (img == VK_NULL_HANDLE)
                    return;

                imageBarrier(cmd, img,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            };

            initNrdOutImage(nrdFrameImages[i].diffImage);
            initNrdOutImage(nrdFrameImages[i].specImage);
            initNrdOutImage(nrdFrameImages[i].validationImage);

            nrdOutputsInitialized[i] = true;
        }
        else if (nrdOutputsSampled[i])
        {
            auto makeWritable = [&](VkImage img)
            {
                if (img == VK_NULL_HANDLE)
                    return;

                imageBarrier(cmd, img,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            };

            makeWritable(nrdFrameImages[i].diffImage);
            makeWritable(nrdFrameImages[i].specImage);
            makeWritable(nrdFrameImages[i].validationImage);
            nrdOutputsSampled[i] = false;
        }
    }


    if (hasDedicatedTransfer && computeFamily != transferFamily)
    {
        std::vector<VkBufferMemoryBarrier> acqs;
        acqs.reserve(5);
        auto addAcq = [&](VkBuffer b)
        {
            if (!b) return;
            VkBufferMemoryBarrier bb{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
            bb.srcAccessMask = 0;
            bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bb.srcQueueFamilyIndex = transferFamily;
            bb.dstQueueFamilyIndex = computeFamily;
            bb.buffer = b;
            bb.offset = 0;
            bb.size   = VK_WHOLE_SIZE;
            acqs.push_back(bb);
        };
        addAcq(tlasNodesBuf[i]);
        addAcq(tlasInstBuf[i]);
        addAcq(tlasIdxBuf[i]);
        addAcq(prevTlasInstBuf[i]);
        addAcq(prevTlasIdxBuf[i]);

        if (!acqs.empty())
        {
            vkCmdPipelineBarrier(computeCommandBuffers[i],
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(acqs.size()), acqs.data(), 0, nullptr);
        }
    }
    else
    {
        std::vector<VkBufferMemoryBarrier> vis;
        vis.reserve(5);
        auto addVis = [&](VkBuffer b)
        {
            if (!b) return;
            VkBufferMemoryBarrier bb{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
            bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.buffer = b;
            bb.offset = 0;
            bb.size   = VK_WHOLE_SIZE;
            vis.push_back(bb);
        };
        addVis(tlasNodesBuf[i]);
        addVis(tlasInstBuf[i]);
        addVis(tlasIdxBuf[i]);
        addVis(prevTlasInstBuf[i]);
        addVis(prevTlasIdxBuf[i]);

        if (!vis.empty())
        {
            vkCmdPipelineBarrier(computeCommandBuffers[i],
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(vis.size()), vis.data(), 0, nullptr);
        }
    }

    

    uint32_t gx = (rayTraceExtent.width  + 7)/8;
    uint32_t gy = (rayTraceExtent.height + 7)/8;

    if constexpr (SimpleRayTrace)
    {
        VkDescriptorSet sets[3] = {
            computeStaticSet,
            computeFrameSets[i],
            computeDynamicSets[i]
        };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout,
                                0,
                                3,
                                sets,
                                0, nullptr);

        vkCmdDispatch(cmd, gx, gy, 1);

        imageBarrier(cmd, offscreenImage[i],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    }
    else
    {
        // Ensure NRD input images are writable this frame (last frame they were sampled by NRD)
        if (nrdInputsSampled[i])
        {
            auto transitionToGeneral = [&](VkImage img)
            {
                if (img == VK_NULL_HANDLE)
                    return;
                imageBarrier(cmd, img,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
            };
            transitionToGeneral(nrdInputs[i].diffRadianceHit.image);
            transitionToGeneral(nrdInputs[i].specRadianceHit.image);
            transitionToGeneral(nrdInputs[i].normalRoughness.image);
            transitionToGeneral(nrdInputs[i].viewZ.image);
            transitionToGeneral(nrdInputs[i].motionVec.image);
            nrdInputsSampled[i] = false;
        }

        VkDescriptorSet sets[3] = {
            computeStaticSet,
            computeFrameSets[i],
            computeDynamicSets[i]
        };

        auto bindAndDispatch = [&](VkPipeline pipeline, uint32_t gx, uint32_t gy)
        {
            if (pipeline == VK_NULL_HANDLE)
                return;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    computePipelineLayout,
                                    0,
                                    3,
                                    sets,
                                    0, nullptr);
            vkCmdDispatch(cmd, gx, gy, 1);
        };

        VkMemoryBarrier memBarrier{};
        memBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        auto passBarrier = [&]()
        {
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0,
                                 1, &memBarrier,
                                 0, nullptr,
                                 0, nullptr);
        };

        const uint32_t maxPaths = getMaxPaths();
        const VkDeviceSize pathCount = VkDeviceSize(maxPaths);

        VkDeviceSize bufferSize = pathCount * sizeof(PathHeader);
        VkDeviceSize bufferSize2 = pathCount * sizeof(PixelStatsGPU);

        vkCmdFillBuffer(cmd, pixelStatsBuf[i], 0, bufferSize2, 0u);
        // Primary path initialization: new path, extend, write NRD inputs
        bindAndDispatch(rayTracePrimaryNewPathPipeline, gx, gy);
        passBarrier();
        bindAndDispatch(rayTracePrimaryExtendRayPipeline, gx, gy);
        passBarrier();
        bindAndDispatch(rayTraceWritePrimaryNRDPipeline, gx, gy);
        passBarrier();
        
        vkCmdFillBuffer(cmd, pathHeaderBuf[i], 0, bufferSize, 0u);
        
        for (uint32_t bounce = 0; bounce < 32; bounce++)
        {
            bindAndDispatch(rayTraceMaterialDiffusePipeline, gx, gy);
            passBarrier();

            bindAndDispatch(rayTraceShadowRayPipeline, gx, gy);
            passBarrier();

            bindAndDispatch(rayTraceLogicDiffusePipeline, gx, gy);
            passBarrier();

            bindAndDispatch(rayTraceExtendRayPipeline, gx, gy);
            passBarrier();
        }
        
        vkCmdFillBuffer(cmd, pathHeaderBuf[i], 0, bufferSize, 0u);

        for (uint32_t bounce = 0; bounce < 32; bounce++)
        {
            bindAndDispatch(rayTraceMaterialSpecularPipeline, gx, gy);
            passBarrier();

            bindAndDispatch(rayTraceShadowRayPipeline, gx, gy);
            passBarrier();
            
            bindAndDispatch(rayTraceLogicSpecularPipeline, gx, gy);
            passBarrier();

            bindAndDispatch(rayTraceExtendRayPipeline, gx, gy);
            passBarrier();

        }

        
        transitionNrdInputsForDenoising(cmd, i);
        runNRD(cmd, i);
        passBarrier();

        bindAndDispatch(rayTraceFinalWritePipeline, gx, gy);
        passBarrier();
        transitionNrdInputsForDenoising(cmd, i);
        

        // Prepare offscreen for sampling in FSR
        imageBarrier(cmd, offscreenImage[i],
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);

        bindAndDispatch(FSRPipeline, (swapChainExtent.width  + 15)/16, (swapChainExtent.height + 15)/16);
        passBarrier();

        bindAndDispatch(FSRSharpenPipeline, (swapChainExtent.width  + 15)/16, (swapChainExtent.height + 15)/16);
        passBarrier();

        // Copy current viewZ into prevViewZ image for next frame motion vectors
        VkImageMemoryBarrier copyBarriers[2]{};
        copyBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        copyBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        copyBarriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        copyBarriers[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        copyBarriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copyBarriers[0].image = nrdInputs[i].viewZ.image;
        copyBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        copyBarriers[1] = copyBarriers[0];
        copyBarriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copyBarriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        copyBarriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copyBarriers[1].image = prevViewZImage[i];

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             2, copyBarriers);

        VkImageCopy copy{};
        copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.extent = { rayTraceExtent.width, rayTraceExtent.height, 1 };
        vkCmdCopyImage(cmd,
                       nrdInputs[i].viewZ.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       prevViewZImage[i],        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &copy);

        // Return layouts: keep NRD input viewZ as sampled, prevViewZ as writable
        copyBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        copyBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        copyBarriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyBarriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        copyBarriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copyBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        copyBarriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copyBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             2, copyBarriers);

        nrdInputsSampled[i] = true;
    }
    

    if constexpr (!SimpleRayTrace)
    {
        imageBarrier(cmd, fsrImage[i],
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    }



    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, qpCmpEnd(i));
    
    vkEndCommandBuffer(cmd);
}

void ShowcaseApp::recordGraphicsCommands(uint32_t frameIndex, uint32_t swapImageIndex)
{
    VkCommandBuffer cmd = graphicsCommandBuffers[frameIndex];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo cmdBufferBeginInfo{};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &cmdBufferBeginInfo);

    vkCmdResetQueryPool(cmd, queryPool, qpGfxBegin(frameIndex), 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,    queryPool, qpGfxBegin(frameIndex));
    
    VkClearValue clear{};
    clear.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = swapChainFramebuffers[swapImageIndex];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = swapChainExtent;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsSets[frameIndex], 0, nullptr);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, qpGfxEnd(frameIndex));

    vkEndCommandBuffer(cmd);
}

void ShowcaseApp::destroyCommandPoolAndBuffers()
{
    if (graphicsCommandPool)
    {
        vkDestroyCommandPool(device, graphicsCommandPool, nullptr);
        graphicsCommandPool = VK_NULL_HANDLE;
    }
    if (computeCommandPool)
    {
        vkDestroyCommandPool(device, computeCommandPool, nullptr);
        computeCommandPool = VK_NULL_HANDLE;
    }
}



void ShowcaseApp::frameTLASPrepare(uint32_t frameIndex)
{
    std::vector<TLASNodeGPU> nodes;
    std::vector<InstanceDataGPU> instances;
    updateInstances(ShowcaseApp::instances, scene);
    topLevelAS = buildTLAS(ShowcaseApp::instances);

    for (auto& node : topLevelAS.nodes)
        nodes.push_back(packNode(node));
    for (int i = 0; i < topLevelAS.instances.size(); i++)
    {
        auto& instance = topLevelAS.instances[i];
        AABB modelBB = scene.objects[i].boundingBox;
        instances.push_back(packInstance(instance, modelBB));
    }
    uploadTLASForFrame(frameIndex, nodes, instances, topLevelAS.instanceIndices);
}

void ShowcaseApp::update(float dt)
{
    const float speed = 5.0f;
    glm::vec3 U, V, W;
    scene.camera.basis(U, V, W);

    glm::vec3 worldUp(0,1,0);
    glm::vec3 fwd = -W;
    fwd -= glm::dot(fwd, worldUp) * worldUp;
    if (glm::length(fwd) > 0.0f) fwd = glm::normalize(fwd);

    glm::vec3 right = glm::normalize(glm::cross(fwd, worldUp));

    glm::vec3 move(0.0f);

    if (glfwGetKey(window.handle(), GLFW_KEY_W) == GLFW_PRESS) move += fwd;
    if (glfwGetKey(window.handle(), GLFW_KEY_S) == GLFW_PRESS) move -= fwd;
    if (glfwGetKey(window.handle(), GLFW_KEY_A) == GLFW_PRESS) move -= right;
    if (glfwGetKey(window.handle(), GLFW_KEY_D) == GLFW_PRESS) move += right;

    if (glfwGetKey(window.handle(), GLFW_KEY_SPACE) == GLFW_PRESS)       move += worldUp;
    if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window.handle(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) move -= worldUp;

    if (glm::length(move) > 0.0f)
    {
        move = glm::normalize(move) * (speed * dt);
        scene.camera.position += move;
        scene.camera.target   += move;
        hasMoved = true;
    }


    if (selectedInstance >= 0 && selectedInstance < static_cast<int>(scene.objects.size()))
    {
        float objectGeneralSpeed = scene.objects[selectedInstance].animation.SpeedScalar;
        if (objectGeneralSpeed == 0.0f)
            objectGeneralSpeed = 1.0f;
        glm::vec3 objMove(0.0f);

        if (glfwGetKey(window.handle(), GLFW_KEY_F) == GLFW_PRESS)
            objMove -= right;
        if (glfwGetKey(window.handle(), GLFW_KEY_G) == GLFW_PRESS)
            objMove += right;
        if (glfwGetKey(window.handle(), GLFW_KEY_H) == GLFW_PRESS)
            objMove += worldUp;
        if (glfwGetKey(window.handle(), GLFW_KEY_J) == GLFW_PRESS)
            objMove -= worldUp;
        if (glfwGetKey(window.handle(), GLFW_KEY_K) == GLFW_PRESS)
            objMove += fwd;
        if (glfwGetKey(window.handle(), GLFW_KEY_L) == GLFW_PRESS)
            objMove -= fwd;

        if (glm::length(objMove) > 0.0f)
        {
            objMove = glm::normalize(objMove) * (objectGeneralSpeed * dt);
            scene.objects[selectedInstance].transform.translate += objMove;
        }
        const float rotSpeed = 45.0f * objectGeneralSpeed; // degrees per second (tweak as you like)

        Transform& tr = scene.objects[selectedInstance].transform;

        if (glfwGetKey(window.handle(), GLFW_KEY_T) == GLFW_PRESS)
            tr.rotation.x += rotSpeed * dt;
        if (glfwGetKey(window.handle(), GLFW_KEY_Y) == GLFW_PRESS)
            tr.rotation.x -= rotSpeed * dt;
        if (glfwGetKey(window.handle(), GLFW_KEY_U) == GLFW_PRESS)
            tr.rotation.y += rotSpeed * dt;
        if (glfwGetKey(window.handle(), GLFW_KEY_I) == GLFW_PRESS)
            tr.rotation.y -= rotSpeed * dt;
        if (glfwGetKey(window.handle(), GLFW_KEY_O) == GLFW_PRESS)
            tr.rotation.z += rotSpeed * dt;
        if (glfwGetKey(window.handle(), GLFW_KEY_P) == GLFW_PRESS)
            tr.rotation.z -= rotSpeed * dt;
        auto wrapDeg = [](float a)
        {
            a = std::fmod(a, 360.0f);
            if (a < 0.0f)
                a += 360.0f;
            return a;
        };

        tr.rotation.x = wrapDeg(tr.rotation.x);
        tr.rotation.y = wrapDeg(tr.rotation.y);
        tr.rotation.z = wrapDeg(tr.rotation.z);
    }

    static bool prevB      = false;
    static bool prevRight  = false;
    static bool prevLeft   = false;
    static bool prevEsc    = false;
    static bool prevDebugN = false;
    static bool prevDebugM = false;

    int bState      = glfwGetKey(window.handle(), GLFW_KEY_B);
    int rightState  = glfwGetKey(window.handle(), GLFW_KEY_PERIOD);
    int leftState   = glfwGetKey(window.handle(), GLFW_KEY_COMMA);
    int escState    = glfwGetKey(window.handle(), GLFW_KEY_ESCAPE);
    int debugNState = glfwGetKey(window.handle(), GLFW_KEY_N);
    int debugMState = glfwGetKey(window.handle(), GLFW_KEY_M);

    bool bPressed      = (bState      == GLFW_PRESS && !prevB);
    bool rightPressed  = (rightState  == GLFW_PRESS && !prevRight);
    bool leftPressed   = (leftState   == GLFW_PRESS && !prevLeft);
    bool escPressed    = (escState    == GLFW_PRESS && !prevEsc);
    bool debugNPressed = (debugNState == GLFW_PRESS && !prevDebugN);
    bool debugMPressed = (debugMState == GLFW_PRESS && !prevDebugM);

    prevB      = (bState      == GLFW_PRESS);
    prevRight  = (rightState  == GLFW_PRESS);
    prevLeft   = (leftState   == GLFW_PRESS);
    prevEsc    = (escState    == GLFW_PRESS);
    prevDebugN = (debugNState == GLFW_PRESS);
    prevDebugM = (debugMState == GLFW_PRESS);

    if (!bTransitionActive && bPressed)
    {
        viewFaces = !viewFaces;

        bTransitionActive = true;
        bInterpTime       = 0.0f;
    }

    bInterpInt = 0;

    if (bTransitionActive)
    {
        bInterpTime += dt;

        float t = bInterpTime / bTransitionDuration;
        if (t >= 1.0f)
        {
            t = 1.0f;
            bTransitionActive = false;
        }
        else
        {
            t = glm::clamp(t, 0.0f, 1.0f);
            bInterpInt = static_cast<uint32_t>(t * B_INTERP_MAX + 0.5f);
            if (bInterpInt == 0)
                bInterpInt = 1;
        }
    }

    if (selectedInstance >= 0)
        if (selectedInstance >= int(topLevelAS.instances.size()) || selectedInstance >= 16)
            selectedInstance = -1;

    if (escPressed)
        selectedInstance = -1;

    if (debugNPressed)
    {
        debugNrdValidation = !debugNrdValidation;
        if (debugNrdValidation)
            debugNrdInputs = false;
    }

    if (debugMPressed)
    {
        debugNrdInputs = !debugNrdInputs;
        if (debugNrdInputs)
            debugNrdValidation = false;
    }

    if (rightPressed)
    {
        if (topLevelAS.instances.empty())
            selectedInstance = -1;
        else if (selectedInstance < 0)
            selectedInstance = 0;
        else if (selectedInstance + 1 < int(topLevelAS.instances.size()) && selectedInstance + 1 < 16)
            ++selectedInstance;
        else if (selectedInstance + 1 >= int(topLevelAS.instances.size()))
            selectedInstance = -1;
    }

    if (leftPressed)
    {
        if (selectedInstance > 0)
            --selectedInstance;
        else if (selectedInstance == 0)
            selectedInstance = -1;
    }
}

void ShowcaseApp::updateFsrConstants(uint32_t frameIndex)
{
    float inputViewportW = float(rayTraceExtent.width);
    float inputViewportH = float(rayTraceExtent.height);

    float inputSizeW = float(rayTraceExtent.width);
    float inputSizeH = float(rayTraceExtent.height);

    float outputW = float(swapChainExtent.width);
    float outputH = float(swapChainExtent.height);

    AU4 c0, c1, c2, c3, c4;

    FsrEasuCon(
        c0, c1, c2, c3,
        inputViewportW, inputViewportH,
        inputSizeW,     inputSizeH,
        outputW,        outputH);

    FsrConstants data{};
    memcpy(data.con0, c0, sizeof(uint32_t) * 4);
    memcpy(data.con1, c1, sizeof(uint32_t) * 4);
    memcpy(data.con2, c2, sizeof(uint32_t) * 4);
    memcpy(data.con3, c3, sizeof(uint32_t) * 4);

    FsrRcasCon(c4, 0.0f);
    memcpy(data.conSharp, c4, sizeof(uint32_t) * 4);
    
    memcpy(fsrConstMapped[frameIndex], &data, sizeof(FsrConstants));
}

void ShowcaseApp::updateNRDCommonSettings(float dt)
{
    nrd::CommonSettings common{};

    // 1) Motion vectors: current shader writes delta in PIXELS.
    // NRD expects: mv = IN_MV * motionVectorScale; pixelUvPrev = pixelUv + mv.xy
    // So: motionVectorScale = (1 / resolution) to convert pixels -> UV.
    uint16_t currResW = static_cast<uint16_t>(rayTraceExtent.width);
    uint16_t currResH = static_cast<uint16_t>(rayTraceExtent.height);

    common.isMotionVectorInWorldSpace = false;
    // NRD expects jitter in pixel units in range [-0.5, 0.5]
    common.cameraJitter[0]     = currJitterPx.x;
    common.cameraJitter[1]     = currJitterPx.y;
    common.cameraJitterPrev[0] = prevJitterPx.x;
    common.cameraJitterPrev[1] = prevJitterPx.y;

    // 2) Matrices (non-jittered!)
    auto copyMat = [](const Mat4& src, float dst[16]) {
        std::memcpy(dst, src.data(), sizeof(float) * 16);
    };

    // Current
    copyMat(currProj, common.viewToClipMatrix);
    copyMat(currView, common.worldToViewMatrix);

    // Previous: if no valid history yet, fall back to current
    bool hasPrev = !(lastViewProjInv[0][0] == 0.0f &&
                     lastViewProjInv[1][1] == 0.0f &&
                     lastViewProjInv[2][2] == 0.0f);

    if (hasPrev)
    {
        copyMat(lastProj, common.viewToClipMatrixPrev);
        copyMat(lastView, common.worldToViewMatrixPrev);
    }
    else
    {
        copyMat(currProj, common.viewToClipMatrixPrev);
        copyMat(currView, common.worldToViewMatrixPrev);
    }

    // worldPrevToWorldMatrix: identity (no special virtual normal space)
    // common.worldPrevToWorldMatrix already defaults to identity in the header initializer.

    // 3) Resolution / rect
    common.resourceSize[0] = currResW;
    common.resourceSize[1] = currResH;
    common.rectSize[0]     = currResW;
    common.rectSize[1]     = currResH;

    uint16_t prevW     = nrdResourceSizePrev[0];
    uint16_t prevH     = nrdResourceSizePrev[1];
    uint16_t prevRectW = nrdRectSizePrev[0];
    uint16_t prevRectH = nrdRectSizePrev[1];

    if (prevW == 0 || prevH == 0) {
        prevW = currResW;
        prevH = currResH;
    }
    if (prevRectW == 0 || prevRectH == 0) {
        prevRectW = currResW;
        prevRectH = currResH;
    }

    common.resourceSizePrev[0] = prevW;
    common.resourceSizePrev[1] = prevH;
    common.rectSizePrev[0]     = prevRectW;
    common.rectSizePrev[1]     = prevRectH;

    // 4) Depth / range
    // Your IN_VIEWZ is world distance (float, r32), so:
    common.viewZScale = 1.0f;

    // ---- 2.5D motion scaling ----
    // IN_MV.xy is in *pixels*; NRD wants UV-scale: mv.xy * (1/w, 1/h)
    common.motionVectorScale[0] = 1.0f / float(currResW);
    common.motionVectorScale[1] = 1.0f / float(currResH);

    // IN_MV.z is (viewZprev - viewZ) in same units as IN_VIEWZ.
    // If viewZScale != 1 in the future, this should be 1/viewZScale.
    common.motionVectorScale[2] = 1.0f;    // non-zero => 2.5D

    // This must be a monotonically increasing frame counter
    common.frameIndex       = nrdFrameIndex++;
    common.accumulationMode = nrd::AccumulationMode::CONTINUE;
    common.denoisingRange = 60000.0f;

    // Disocclusion + range – tweak these later
    common.disocclusionThreshold   = 0.01f;
    common.disocclusionThresholdAlternate = 0.05f;

    common.enableValidation = true;

    nrdIntegration.SetCommonSettings(common);

    // Track prev sizes for next frame
    nrdResourceSizePrev[0] = currResW;
    nrdResourceSizePrev[1] = currResH;
    nrdRectSizePrev[0]     = currResW;
    nrdRectSizePrev[1]     = currResH;
}


void ShowcaseApp::transitionNrdOutputsForSampling(VkCommandBuffer cmd, uint32_t frameIndex)
{
    const NrdFrameImage& frame = nrdFrameImages[frameIndex];

    VkImageMemoryBarrier2 barriers[3]{};

    // Diffuse (NRD OUT_DIFF_RADIANCE_HITDIST)
    barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[0].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].image         = frame.diffImage;
    barriers[0].subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[0].subresourceRange.baseMipLevel   = 0;
    barriers[0].subresourceRange.levelCount     = 1;
    barriers[0].subresourceRange.baseArrayLayer = 0;
    barriers[0].subresourceRange.layerCount     = 1;

    barriers[1]               = barriers[0];
    barriers[1].image         = frame.specImage;

    barriers[2]               = barriers[0];
    barriers[2].image         = frame.validationImage;

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 3;
    dep.pImageMemoryBarriers    = barriers;

    vkCmdPipelineBarrier2(cmd, &dep);
    nrdOutputsSampled[frameIndex] = true;
    auto transitionFromShaderReadToGeneral = [&](VkImage img)
    {
        if (img == VK_NULL_HANDLE)
            return;

        imageBarrier(cmd, img,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED);
    };

    transitionFromShaderReadToGeneral(nrdInputs[frameIndex].diffRadianceHit.image);
    transitionFromShaderReadToGeneral(nrdInputs[frameIndex].specRadianceHit.image);
    transitionFromShaderReadToGeneral(nrdInputs[frameIndex].normalRoughness.image);
    transitionFromShaderReadToGeneral(nrdInputs[frameIndex].viewZ.image);
    transitionFromShaderReadToGeneral(nrdInputs[frameIndex].motionVec.image);
}

void ShowcaseApp::runNRD(VkCommandBuffer cmd, uint32_t frameIndex)
{
    // 1) Tell NRD it's a new frame
    nrdIntegration.NewFrame();

    // 2) Common settings (resolution, accumulation, frame index, etc.)
    updateNRDCommonSettings(dt);

    nrd::RelaxSettings relax{};
    relax.enableAntiFirefly = true;
    nrdIntegration.SetDenoiserSettings(nrdRelaxId, &relax);

    // 4) Build ResourceSnapshot with IN_* and OUT_* resources
    nrd::ResourceSnapshot snapshot{};

    auto makeResourceVK = [](VkImage image, VkFormat format, nri::Layout layout, nri::AccessBits access) -> nrd::Resource
    {
        nrd::Resource r{};

        r.vk.image  = reinterpret_cast<VKNonDispatchableHandle>(image);
        r.vk.format = static_cast<VKEnum>(format);

        r.state.access = access;
        r.state.layout = layout;
        r.state.stages = nri::StageBits::COMPUTE_SHADER;

        return r;
    };
    // ---------- IN_ resources ----------
    nrd::Resource inDiff   = makeResourceVK(nrdInputs[frameIndex].diffRadianceHit.image,  nrdInputs[frameIndex].diffRadianceHit.format, nri::Layout::SHADER_RESOURCE, nri::AccessBits::SHADER_RESOURCE);
    nrd::Resource inSpec   = makeResourceVK(nrdInputs[frameIndex].specRadianceHit.image,  nrdInputs[frameIndex].specRadianceHit.format, nri::Layout::SHADER_RESOURCE, nri::AccessBits::SHADER_RESOURCE);
    nrd::Resource inNR     = makeResourceVK(nrdInputs[frameIndex].normalRoughness.image,  nrdInputs[frameIndex].normalRoughness.format, nri::Layout::SHADER_RESOURCE, nri::AccessBits::SHADER_RESOURCE);
    nrd::Resource inViewZ  = makeResourceVK(nrdInputs[frameIndex].viewZ.image,            nrdInputs[frameIndex].viewZ.format,           nri::Layout::SHADER_RESOURCE, nri::AccessBits::SHADER_RESOURCE);
    nrd::Resource inMotion = makeResourceVK(nrdInputs[frameIndex].motionVec.image,        nrdInputs[frameIndex].motionVec.format,       nri::Layout::SHADER_RESOURCE, nri::AccessBits::SHADER_RESOURCE);

    snapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, inDiff);
    snapshot.SetResource(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, inSpec);
    snapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS,      inNR);
    snapshot.SetResource(nrd::ResourceType::IN_VIEWZ,                 inViewZ);
    snapshot.SetResource(nrd::ResourceType::IN_MV,                    inMotion);

    // ---------- OUT_ resources ----------
    nrd::Resource outDiff = makeResourceVK(nrdFrameImages[frameIndex].diffImage, VK_FORMAT_R16G16B16A16_SFLOAT, nri::Layout::GENERAL, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    nrd::Resource outSpec = makeResourceVK(nrdFrameImages[frameIndex].specImage, VK_FORMAT_R16G16B16A16_SFLOAT, nri::Layout::GENERAL, nri::AccessBits::SHADER_RESOURCE_STORAGE);
    nrd::Resource outValidation = makeResourceVK(nrdFrameImages[frameIndex].validationImage, VK_FORMAT_R16G16B16A16_SFLOAT, nri::Layout::GENERAL, nri::AccessBits::SHADER_RESOURCE_STORAGE);

    snapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, outDiff);
    snapshot.SetResource(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, outSpec);
    snapshot.SetResource(nrd::ResourceType::OUT_VALIDATION, outValidation);

    // We manage final Vulkan resource states ourselves; no need for NRD to restore.
    snapshot.restoreInitialState = false;

    // 5) Wrap the VkCommandBuffer into NRI's CommandBufferVKDesc and denoise
    nri::CommandBufferVKDesc cbDesc{};
    cbDesc.vkCommandBuffer = cmd;
    cbDesc.queueType       = nri::QueueType::COMPUTE;

    const nrd::Identifier denoisers[] = { nrdRelaxId };

    nrdIntegration.DenoiseVK(denoisers, 1, cbDesc, snapshot);
    transitionNrdOutputsForSampling(cmd, frameIndex);
}

void ShowcaseApp::run()
{
    createCommandPoolAndBuffers();
    uploadStaticData();

    using clock = std::chrono::high_resolution_clock;

    auto prevTime = clock::now();

    double fps = 0.0;
    double avgMs = 0.0;
    const int history = 100;
    std::array<double, history> samples{};
    int cursor = 0;

    while (!window.shouldClose())
    {
        auto t0 = clock::now();
        glfwPollEvents();
        dt = std::chrono::duration<float>(t0 - prevTime).count();
        prevTime = t0;
        if (dt > 0.1f)
            dt = 0.1f;
        update(dt);

        if (window.wasWindowResized())
            recreateSwapchain();

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex = 0;
        vkResetFences(device, 1, &imageAcquiredFences[currentFrame]);
        VkResult acq = vkAcquireNextImageKHR(
            device, swapChain, UINT64_MAX,
            VK_NULL_HANDLE,
            imageAcquiredFences[currentFrame],
            &imageIndex
        );
        if (acq == VK_ERROR_OUT_OF_DATE_KHR)
        { 
            recreateSwapchain();
            continue;
        }
        if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("failed to acquire swapchain image");
        vkWaitForFences(device, 1, &imageAcquiredFences[currentFrame], VK_TRUE, UINT64_MAX);

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        vkWaitForFences(device, 1, &frameUpload[currentFrame].uploadFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &frameUpload[currentFrame].uploadFence);
        vkWaitForFences(device, 1, &computeFences[currentFrame], VK_TRUE, UINT64_MAX);

        if (queryPool && queryPrimed[currentFrame])
        {
            const uint32_t fi = currentFrame;
            uint64_t ts[4] = {};
            VkResult r = vkGetQueryPoolResults(
                device, queryPool,
                qpBase(fi), 4,
                sizeof(ts), ts, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            
            if (r == VK_SUCCESS)
            {
                const double toMs = timestampPeriodNs / 1.0e6;
                double gfxMs = double(ts[1] - ts[0]) * toMs;
                double cmpMs = double(ts[3] - ts[2]) * toMs;
            
                double combinedMs = cmpMs + gfxMs;
            
                if (FPS) {
                    std::cout << "[GPU] compute: " << cmpMs
                              << " ms, graphics: " << gfxMs
                              << " ms, combined: " << combinedMs << " ms\n";
                }
            }
            queryPrimed[currentFrame] = false;
        }
        vkResetCommandBuffer(frameUpload[currentFrame].uploadCB, 0);
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(frameUpload[currentFrame].uploadCB, &bi);
        frameTLASPrepare(currentFrame);
        vkEndCommandBuffer(frameUpload[currentFrame].uploadCB);

        VkSubmitInfo up{};
        up.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        up.commandBufferCount = 1;
        up.pCommandBuffers = &frameUpload[currentFrame].uploadCB;
        up.signalSemaphoreCount = 1;
        up.pSignalSemaphores = &frameUpload[currentFrame].uploadDone;

        VkQueue uploadQ = (hasDedicatedTransfer ? transferQueue : computeQueue);
        if (vkQueueSubmit(uploadQ, 1, &up, frameUpload[currentFrame].uploadFence) != VK_SUCCESS)
            throw std::runtime_error("Showcase App: failed to subbmit TLAS upload");

        // Update per-frame descriptors after TLAS buffers are (re)allocated.
        updateComputeDescriptor(static_cast<int>(currentFrame));

        uint32_t flags = 0;

        if (viewFaces)
            flags |= FLAG_B;

        if (debugNrdValidation)
            flags |= FLAG_DEBUG_NRD_VALIDATION;

        if (debugNrdInputs)
            flags |= FLAG_DEBUG_NRD_INPUTS;

        if (selectedInstance >= 0)
        {
            flags |= FLAG_INSTANCE_SEL;
            uint32_t idx = static_cast<uint32_t>(selectedInstance) & MASK_INSTANCE_IDX;
            flags |= idx;
        }
        if (bInterpInt > 0)
            flags |= (bInterpInt << B_INTERP_SHIFT) & B_INTERP_MASK;
        
        glm::vec2 jitterPx = getHaltonJitterPx(nrdFrameIndex);
        glm::vec2 jitterUv = jitterPx / glm::vec2(rayTraceExtent.width, rayTraceExtent.height);

        prevJitterPx = currJitterPx;
        prevJitterUV = currJitterUV;
        currJitterPx = jitterPx;
        currJitterUV = jitterUv;
        
        Mat4 currViewProj{};
        Mat4 currViewProjInv{};
        Mat4 view{}, proj{};
        ParamsGPU params = makeParamsForVulkan(
            rayTraceExtent,
            topLevelAS.root,
            dt,
            scene.camera.position,
            scene.camera.target,
            scene.camera.up,
            scene.camera.vfovDeg,
            scene.camera.nearPlane,
            scene.camera.farPlane,
            flags,
            nrdFrameIndex,
            &currViewProj,
            &currViewProjInv,
            &view,
            &proj
        );

        currView = view;
        currProj = proj;
        params.cameraJitter      = currJitterUV;
        params.cameraJitterPrev  = prevJitterUV;

        std::memcpy(paramsMapped[currentFrame], &params, sizeof(ParamsGPU));
        writeParamsBindingForFrame(currentFrame);
        if constexpr (!SimpleRayTrace)
        {
            PrevCamData prev;
            bool hasPrev = !(lastViewProjInv[0][0] == 0.0f && lastViewProjInv[1][1] == 0.0f && lastViewProjInv[2][2] == 0.0f);
            prev.prevV    = hasPrev ? lastView    : currView;
            prev.prevVP   = hasPrev ? lastViewProj : currViewProj;
            prev.currVP   = currViewProj;
            prev.currV    = currView;
            if (prevCamMapped[currentFrame])
                std::memcpy(prevCamMapped[currentFrame], &prev, sizeof(PrevCamData));
        }

        if constexpr (!SimpleRayTrace)
            updateFsrConstants(currentFrame);
        recordComputeCommands(currentFrame);
        recordGraphicsCommands(currentFrame, imageIndex);
        queryPrimed[currentFrame] = true;

        vkResetFences(device, 1, &computeFences[currentFrame]);
        if constexpr (!SimpleRayTrace)
        {
            lastViewProj    = currViewProj;
            lastViewProjInv = currViewProjInv;
            lastView = currView;
            lastProj = currProj;
        }
        
        VkSubmitInfo submitCompute{};
        submitCompute.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSems[1]   = { frameUpload[currentFrame].uploadDone };
        VkPipelineStageFlags waitComputeStages[1] = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
        submitCompute.waitSemaphoreCount = 1;
        submitCompute.pWaitSemaphores    = waitSems;
        submitCompute.pWaitDstStageMask  = waitComputeStages;

        submitCompute.commandBufferCount = 1;
        submitCompute.pCommandBuffers = &computeCommandBuffers[currentFrame];
        submitCompute.signalSemaphoreCount = 1;
        submitCompute.pSignalSemaphores = &computeDone[currentFrame];
        
        if (vkQueueSubmit(computeQueue, 1, &submitCompute, computeFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to submit compute command buffer!");

        VkPipelineStageFlags waitStage[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
        VkSemaphore signalSem[2] = {imageAvailableSemaphores[currentFrame], computeDone[currentFrame]};

        VkPipelineStageFlags waitStages[1] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
        VkSemaphore          waits[1]      = { computeDone[currentFrame] };
        VkSubmitInfo submitGraphics{};
        submitGraphics.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitGraphics.waitSemaphoreCount   = 1;
        submitGraphics.pWaitSemaphores      = waits;
        submitGraphics.pWaitDstStageMask    = waitStages;
        submitGraphics.commandBufferCount   = 1;
        submitGraphics.pCommandBuffers      = &graphicsCommandBuffers[currentFrame];
        submitGraphics.signalSemaphoreCount = 1;
        submitGraphics.pSignalSemaphores    = &imageRenderFinished[imageIndex];

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        if (vkQueueSubmit(graphicsQueue, 1, &submitGraphics, inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to submit graphics command buffer!");

        VkSemaphore renderDone = imageRenderFinished[imageIndex];

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderDone;
        present.swapchainCount = 1;
        present.pSwapchains = &swapChain;
        present.pImageIndices = &imageIndex;

        VkResult pres = vkQueuePresentKHR(presentQueue, &present);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR || window.wasWindowResized())
        {
            recreateSwapchain();
            continue;
        }
        else if (pres != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to present swapchain image!");

        currentFrame = (currentFrame + 1) % SwapChain::MAX_FRAMES_IN_FLIGHT;
        
        if (FPS)
        {
            double ms = double(dt) * 1000.0;
            samples[cursor++ % history] = ms;
            double sum = 0.0; for (double s : samples) sum += s;
            avgMs = sum / history;
            fps = 1000.0 / avgMs;
            std::cout << "[CPU] avg FPS: " << fps << " avg MS: " << avgMs << std::endl;
        }
    }

    vkDeviceWaitIdle(device);
    destroyCommandPoolAndBuffers();
}
