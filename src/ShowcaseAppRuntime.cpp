#include "ShowcaseApp.hpp"
#include <iostream>

// --------- helpers -----------

inline AABB boundsOfRange(const std::vector<InstanceData>& inst, const std::vector<uint32_t>& idx, uint32_t first, uint32_t count)
{
    AABB b;
    for (uint32_t i = 0; i < count; ++i)
    {
        b = merge(b, inst[idx[first + i]].worldAABB);
    }
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
        node.first = first;
        node.count = count;
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

inline TLAS buildTLAS(const std::vector<InstanceData>& instances)
{
    TLAS tlas;
    tlas.instances = instances;

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

inline InstanceDataGPU packInstance(const InstanceData& data)
{
    InstanceDataGPU out{};

    const glm::mat4 M = affineToMat4(data.modelToWorld);
    const glm::mat4 W = affineToMat4(data.worldToModel);

    out.modelToWorld[0] = glm::vec4(M[0][0], M[1][0], M[2][0], M[3][0]);
    out.modelToWorld[1] = glm::vec4(M[0][1], M[1][1], M[2][1], M[3][1]);
    out.modelToWorld[2] = glm::vec4(M[0][2], M[1][2], M[2][2], M[3][2]);

    out.worldToModel[0] = glm::vec4(W[0][0], W[1][0], W[2][0], W[3][0]);
    out.worldToModel[1] = glm::vec4(W[0][1], W[1][1], W[2][1], W[3][1]);
    out.worldToModel[2] = glm::vec4(W[0][2], W[1][2], W[2][2], W[3][2]);

    out.aabbMin = glm::vec4(data.worldAABB.min, 0.0f);
    out.aabbMax = glm::vec4(data.worldAABB.max, 0.0f);

    out.bases0 = glm::uvec4(data.nodeBase, data.triBase, data.shadeTriBase, data.materialBase);
    out.bases1 = glm::uvec4(data.textureBase, 0u, 0u, 0u);
    return out;
}

void ShowcaseApp::ensureBufferCapacity(VkBuffer& buf, VkDeviceMemory& mem, VkDeviceSize neededSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags)
{
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

void ShowcaseApp::uploadBytesToDeviceLocal(const void* src, VkDeviceSize bytes, VkBuffer dstDeviceLocal)
{
    if (bytes == 0) return;

    VkBuffer stageBuf = VK_NULL_HANDLE;
    VkDeviceMemory stageMem = VK_NULL_HANDLE;
    createBuffer(device, physicalDevice, bytes,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stageBuf, stageMem);

    void* p = nullptr;
    vkMapMemory(device, stageMem, 0, bytes, 0, &p);
    std::memcpy(p, src, size_t(bytes));
    vkUnmapMemory(device, stageMem);

    copyBuffer(device, computeCommandPool, computeQueue, stageBuf, dstDeviceLocal, bytes);

    vkDestroyBuffer(device, stageBuf, nullptr);
    vkFreeMemory(device, stageMem, nullptr);
}

void ShowcaseApp::uploadTLASForFrame(uint32_t frameIndex,
                                    const std::vector<TLASNodeGPU>& tlasNodes,
                                    const std::vector<InstanceDataGPU>& tlasInstances,
                                    const std::vector<uint32_t>& instanceIndices)
{
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

    if (nodesBytes)
        uploadBytesToDeviceLocal(tlasNodes.data(), nodesBytes, tlasNodesBuf[frameIndex]);
    if (instBytes) 
        uploadBytesToDeviceLocal(tlasInstances.data(), instBytes, tlasInstBuf[frameIndex]);
    if (idxBytes)
        uploadBytesToDeviceLocal(instanceIndices.data(), idxBytes, tlasIdxBuf[frameIndex]);

    VkDescriptorBufferInfo b6{ tlasNodesBuf[frameIndex], 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b7{ tlasInstBuf[frameIndex],  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo b8{ tlasIdxBuf[frameIndex],   0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; i++)
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    w[0].dstSet = computeSets[frameIndex];
    w[0].dstBinding = 6;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].descriptorCount = 1;
    w[0].pBufferInfo = &b6;

    w[1].dstSet = computeSets[frameIndex];
    w[1].dstBinding = 7;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].descriptorCount = 1;
    w[1].pBufferInfo = &b7;

    w[2].dstSet = computeSets[frameIndex];
    w[2].dstBinding = 8;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].descriptorCount = 1;
    w[2].pBufferInfo = &b8;

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
    
    if (!offscreenInitialized[i])
    {
        imageBarrier(cmd, offscreenImage[i],
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            computeFamily, computeFamily);
        offscreenInitialized[i] = true;
    }
    else
    {
        imageBarrier(cmd, offscreenImage[i],
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            graphicsFamily, computeFamily);
    }
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeSets[i], 0, nullptr);

    uint32_t gx = (swapChainExtent.width  + 7)/8;
    uint32_t gy = (swapChainExtent.height + 7)/8;
    vkCmdDispatch(cmd, gx, gy, 1);

    imageBarrier(cmd, offscreenImage[i],
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        computeFamily, graphicsFamily);

    vkEndCommandBuffer(cmd);
}

void ShowcaseApp::recordGraphicsCommands(uint32_t frameIndex, uint32_t swapImageIndex)
{
    VkCommandBuffer cmd = graphicsCommandBuffers[frameIndex];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo cmdBufferBeginInfo{};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &cmdBufferBeginInfo);

    
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

//temp
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>   // for lookAt, perspective
#include <glm/gtc/type_ptr.hpp>

// Right-handed camera, depth 0..1 (Vulkan)
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
    bool bbview          = false)
{
    const float aspect = float(extent.width) / float(std::max(1u, extent.height));

    // RH view looking from camPos to camTarget
    glm::mat4 view = glm::lookAtRH(camPos, camTarget, camUp);

    // RH projection with depth in [0..1]
    // If you have GLM 0.9.9+: use glm::perspectiveRH_ZO; otherwise glm::perspective does ZO thanks to GLM_FORCE_DEPTH_ZERO_TO_ONE.
    glm::mat4 proj = glm::perspective(glm::radians(fovY_deg), aspect, zNear, zFar);

    // glm::mat4 M = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),  glm::vec3(0,1,0));
    // Vulkan: no automatic Y flip here; your fullscreen pass uses vUV from 0..1 so this is fine.
    glm::mat4 viewProj     = proj * view;
    glm::mat4 viewProjInv  = glm::inverse(viewProj);

    ParamsGPU p{};
    p.viewProjInv = viewProjInv;
    p.camPos_time = glm::vec4(camPos, time);
    p.imageSize   = glm::uvec2(extent.width, extent.height);
    p.rootIndex   = rootIndex;
    p._pad0       = bbview ? 69 : 0;
    return p;
}
//temp

void ShowcaseApp::frameTLASPrepare(uint32_t frameIndex)
{
    std::vector<TLASNodeGPU> nodes;
    std::vector<InstanceDataGPU> instances;
    topLevelAS = buildTLAS(ShowcaseApp::instances);

    for (auto& node : topLevelAS.nodes)
        nodes.push_back(packNode(node));
    for (auto& instance : topLevelAS.instances)
        instances.push_back(packInstance(instance));
    
    uploadTLASForFrame(frameIndex, nodes, instances, topLevelAS.instanceIndices);
}

void ShowcaseApp::update(float dt)
{
    const float speed = 0.1f;
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
    if (glfwGetKey(window.handle(), GLFW_KEY_B) == GLFW_PRESS) bbView = bbView ? false : true;

    // vertical stays consistent: space up, shift down
    if (glfwGetKey(window.handle(), GLFW_KEY_SPACE) == GLFW_PRESS)       move += worldUp;
    if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window.handle(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) move -= worldUp;

    if (glm::length(move) > 0.0f)
    {
        move = glm::normalize(move) * (speed * dt);
        scene.camera.position += move;
        scene.camera.target   += move;
    }
}


void ShowcaseApp::run()
{
    createCommandPoolAndBuffers();
    uploadStaticData();

    while (!window.shouldClose())
    {
        glfwPollEvents();
        update(1.0f);

        if (window.wasWindowResized())
            recreateSwapchain();

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex = 0;
        VkResult acq = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (acq == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapchain();
            continue;
        } 
        else if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Showcase: failed to acquire swapchain image!");

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        frameTLASPrepare(currentFrame);
        //temp
        ParamsGPU params = makeParamsForVulkan(
                            swapChainExtent,
                            /*rootIndex*/ topLevelAS.root,       
                            /*time*/ 0.0f,
                            /*camPos*/ scene.camera.position,
                            /*camTarget*/ scene.camera.target,
                            /*up*/ scene.camera.up,
                            /*fov*/ scene.camera.vfovDeg,
                            /*near*/ scene.camera.nearPlane,
                            /*far*/ scene.camera.farPlane,
                            bbView
                        );
        std::memcpy(paramsMapped[currentFrame], &params, sizeof(ParamsGPU));
        writeParamsBindingForFrame(currentFrame);

        recordComputeCommands(currentFrame);
        recordGraphicsCommands(currentFrame, imageIndex);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        VkSubmitInfo submitCompute{};
        submitCompute.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitCompute.commandBufferCount = 1;
        submitCompute.pCommandBuffers = &computeCommandBuffers[currentFrame];
        submitCompute.signalSemaphoreCount = 1;
        submitCompute.pSignalSemaphores = &computeDone[currentFrame];
        
        if (vkQueueSubmit(computeQueue, 1, &submitCompute, VK_NULL_HANDLE) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to submit compute command buffer!");

        VkPipelineStageFlags waitStage[2] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
        VkSemaphore signalSem[2] = {imageAvailableSemaphores[currentFrame], computeDone[currentFrame]};

        VkSubmitInfo submitGraphics{};
        submitGraphics.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitGraphics.waitSemaphoreCount = 2;
        submitGraphics.pWaitSemaphores = signalSem;
        submitGraphics.pWaitDstStageMask = waitStage;
        submitGraphics.commandBufferCount = 1;
        submitGraphics.pCommandBuffers = &graphicsCommandBuffers[currentFrame];
        submitGraphics.signalSemaphoreCount = 1;
        submitGraphics.pSignalSemaphores = &imageRenderFinished[imageIndex];

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
    }

    vkDeviceWaitIdle(device);
    destroyCommandPoolAndBuffers();
}

