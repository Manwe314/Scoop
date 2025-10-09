#include "ShowcaseApp.hpp"
#include <iostream>

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
    float      zFar      = 2000.0f)
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
    p._pad0       = 0;
    return p;
}
//temp


void ShowcaseApp::run()
{
    createCommandPoolAndBuffers();
    uploadStaticData();

    while (!window.shouldClose())
    {
        glfwPollEvents();

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

        //temp
        ParamsGPU params = makeParamsForVulkan(
                            swapChainExtent,
                            /*rootIndex*/ 0,       
                            /*time*/ 0.0f,
                            /*camPos*/ glm::vec3(0, 0, -10),
                            /*camTarget*/ glm::vec3(0, 0, 0),
                            /*up*/ glm::vec3(0, 1, 0),
                            /*fov*/ 60.0f,
                            /*near*/ 0.1f,
                            /*far*/ 1000.0f
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

