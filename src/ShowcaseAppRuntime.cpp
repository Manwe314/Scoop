#include "ShowcaseApp.hpp"
#include <iostream>



// --------- helpers -----------

constexpr uint32_t FLAG_B            = 1u << 31;
constexpr uint32_t FLAG_INSTANCE_SEL = 1u << 30;
constexpr uint32_t B_INTERP_SHIFT    = 20;
constexpr uint32_t B_INTERP_BITS     = 10;
constexpr uint32_t B_INTERP_MAX      = (1u << B_INTERP_BITS) - 1u;
constexpr uint32_t B_INTERP_MASK     = B_INTERP_MAX << B_INTERP_SHIFT;
constexpr uint32_t MASK_INSTANCE_IDX = 0xFu; 

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
    out.bases1 = glm::uvec4(data.textureBase, 0u, 0u, 0u);
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
                uint32_t flags = 0)
{
    const float aspect = float(extent.width) / float(std::max(1u, extent.height));

    Mat4 view = LookAtRH(camPos, camTarget, camUp);

    Mat4 proj = PerspectiveRH_ZO(glm::radians(fovY_deg), aspect, zNear, zFar);

    proj[1][1] *= -1.0f;
    Mat4 viewProj    = proj * view;
    Mat4 viewProjInv;
    inverse(viewProj, viewProjInv);

    ParamsGPU p{};
    p.viewProjInv = viewProjInv;
    p.camPos_time = glm::vec4(camPos, time);
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
    
    neededSize = nonZero(neededSize, 16);

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

    ensureBufferCapacity(tlasNodesBuf[frameIndex], tlasNodesMem[frameIndex], nodesBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ensureBufferCapacity(tlasInstBuf[frameIndex],  tlasInstMem[frameIndex],  instBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    ensureBufferCapacity(tlasIdxBuf[frameIndex],   tlasIdxMem[frameIndex],   idxBytes,
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
    const VkDeviceSize total    = off;

    ensureUploadStagingCapacity(frameIndex, total);

    if (nodesBytes)
        std::memcpy(static_cast<char*>(frame.mapped) + nodesOff, tlasNodes.data(), size_t(nodesBytes));
    if (instBytes) 
        std::memcpy(static_cast<char*>(frame.mapped) + instOff,  tlasInstances.data(), size_t(instBytes));
    if (idxBytes)
        std::memcpy(static_cast<char*>(frame.mapped) + idxOff,   instanceIndices.data(), size_t(idxBytes));
    

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

        if (!vis.empty())
        {
            vkCmdPipelineBarrier(frame.uploadCB,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(vis.size()), vis.data(), 0, nullptr);
        }
    }

    VkDescriptorBufferInfo b6{ tlasNodesBuf[frameIndex], 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b7{ tlasInstBuf[frameIndex],  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b8{ tlasIdxBuf[frameIndex],   0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i) w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, computeSets[frameIndex], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b6, nullptr };
    w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, computeSets[frameIndex], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b7, nullptr };
    w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, computeSets[frameIndex], 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b8, nullptr };
    vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
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
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
    }

    if (hasDedicatedTransfer && computeFamily != transferFamily)
    {
        std::vector<VkBufferMemoryBarrier> acqs;
        acqs.reserve(3);
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
        vis.reserve(3);
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

        if (!vis.empty())
        {
            vkCmdPipelineBarrier(computeCommandBuffers[i],
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(vis.size()), vis.data(), 0, nullptr);
        }
    }

    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeSets[i], 0, nullptr);

    uint32_t gx = (swapChainExtent.width  + 7)/8;
    uint32_t gy = (swapChainExtent.height + 7)/8;
    vkCmdDispatch(cmd, gx, gy, 1);


    imageBarrier(cmd, offscreenImage[i],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);



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

    static bool prevB     = false;
    static bool prevRight = false;
    static bool prevLeft  = false;
    static bool prevEsc   = false;

    int bState     = glfwGetKey(window.handle(), GLFW_KEY_B);
    int rightState = glfwGetKey(window.handle(), GLFW_KEY_PERIOD);
    int leftState  = glfwGetKey(window.handle(), GLFW_KEY_COMMA);
    int escState   = glfwGetKey(window.handle(), GLFW_KEY_ESCAPE);

    bool bPressed     = (bState     == GLFW_PRESS && !prevB);
    bool rightPressed = (rightState == GLFW_PRESS && !prevRight);
    bool leftPressed  = (leftState  == GLFW_PRESS && !prevLeft);
    bool escPressed   = (escState   == GLFW_PRESS && !prevEsc);

    prevB     = (bState     == GLFW_PRESS);
    prevRight = (rightState == GLFW_PRESS);
    prevLeft  = (leftState  == GLFW_PRESS);
    prevEsc   = (escState   == GLFW_PRESS);

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
        float dt = std::chrono::duration<float>(t0 - prevTime).count();
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

        uint32_t flags = 0;

        if (viewFaces)
            flags |= FLAG_B;

        if (selectedInstance >= 0)
        {
            flags |= FLAG_INSTANCE_SEL;
            uint32_t idx = static_cast<uint32_t>(selectedInstance) & MASK_INSTANCE_IDX;
            flags |= idx;
        }
        if (bInterpInt > 0)
            flags |= (bInterpInt << B_INTERP_SHIFT) & B_INTERP_MASK;
        
        ParamsGPU params = makeParamsForVulkan(
                            swapChainExtent,
                            /*rootIndex*/ topLevelAS.root,       
                            /*time*/ dt,
                            /*camPos*/ scene.camera.position,
                            /*camTarget*/ scene.camera.target,
                            /*up*/ scene.camera.up,
                            /*fov*/ scene.camera.vfovDeg,
                            /*near*/ scene.camera.nearPlane,
                            /*far*/ scene.camera.farPlane,
                            flags
                        );
        std::memcpy(paramsMapped[currentFrame], &params, sizeof(ParamsGPU));
        writeParamsBindingForFrame(currentFrame);

        recordComputeCommands(currentFrame);
        recordGraphicsCommands(currentFrame, imageIndex);
        queryPrimed[currentFrame] = true;

        vkResetFences(device, 1, &computeFences[currentFrame]);
        
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

