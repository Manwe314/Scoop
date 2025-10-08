#include "ShowcaseApp.hpp"
#include <iostream>

void ShowcaseApp::createCommandPoolAndBuffers()
{
    QueueFamiliyIndies indices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.queueFamilyIndex = indices.graphicsFamily.value();
    createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create command pool!");

    commandBuffers.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocateInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate command buffers!");
}

void ShowcaseApp::imageBarrier(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = srcAccess;
    memoryBarrier.dstAccessMask = dstAccess;
    memoryBarrier.oldLayout = oldLayout;
    memoryBarrier.newLayout = newLayout;
    memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.image = image;
    memoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    memoryBarrier.subresourceRange.baseMipLevel = 0;
    memoryBarrier.subresourceRange.levelCount = 1;
    memoryBarrier.subresourceRange.baseArrayLayer = 0;
    memoryBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &memoryBarrier);
}

void ShowcaseApp::recordFrameCommands(VkCommandBuffer cmd, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo cmdBufferBeginInfo{};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &cmdBufferBeginInfo);

    if (!offscreenInitialized)
    {
        imageBarrier(cmd, offscreenImage,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        offscreenInitialized = true;
    }
    else
    {
        imageBarrier(cmd, offscreenImage,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeSet, 0, nullptr);

    uint32_t gx = (swapChainExtent.width  + 7) / 8;
    uint32_t gy = (swapChainExtent.height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    imageBarrier(cmd, offscreenImage,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    VkClearValue clear{};
    clear.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassBeginInfo.renderArea.offset = {0, 0};
    renderPassBeginInfo.renderArea.extent = swapChainExtent;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsSet, 0, nullptr);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);

    vkEndCommandBuffer(cmd);
}

void ShowcaseApp::destroyCommandPoolAndBuffers()
{
    if (commandPool)
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
}

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


        ParamsGPU params = makeDefaultParams(swapChainExtent, 0, 0.0f, glm::vec3(0));
        std::memcpy(paramsMapped[currentFrame], &params, sizeof(ParamsGPU));
        writeParamsBindingForFrame(currentFrame);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordFrameCommands(commandBuffers[currentFrame], imageIndex);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSem = imageRenderFinished[imageIndex];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &signalSem;

        if (vkQueueSubmit(graphicsQueue, 1, &submit, inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to submit draw command buffer!");

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &signalSem;
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

