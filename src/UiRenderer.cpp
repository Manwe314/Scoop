#include "UiRenderer.hpp"


static const UiRenderer::QuadVertex kQuadStrip[4] = {
    {{-0.5f, -0.5f}}, {{ 0.5f, -0.5f}}, {{-0.5f,  0.5f}}, {{ 0.5f,  0.5f}}
};


UiRenderer::UiRenderer(Device& device, uint32_t maxInstances)
: device(device), maxInstanceCount(maxInstances) {
    createQuadBuffer();                 // binding 0
    createInstanceBuffer(maxInstances); // binding 1
}

UiRenderer::~UiRenderer() {
    if (instanceMapped) { vkUnmapMemory(device.device(), instanceMem); instanceMapped = nullptr; }
    if (instanceVB) { vkDestroyBuffer(device.device(), instanceVB, nullptr); instanceVB = VK_NULL_HANDLE; }
    if (instanceMem){ vkFreeMemory(device.device(), instanceMem, nullptr); instanceMem = VK_NULL_HANDLE; }

    if (quadVB) { vkDestroyBuffer(device.device(), quadVB, nullptr); quadVB = VK_NULL_HANDLE; }
    if (quadMem){ vkFreeMemory(device.device(), quadMem, nullptr); quadMem = VK_NULL_HANDLE; }
}

void UiRenderer::createQuadBuffer() {
    VkDeviceSize sz = sizeof(kQuadStrip);

    device.createBuffer(
        sz,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        quadVB, quadMem
    );

    void* data = nullptr;
    vkMapMemory(device.device(), quadMem, 0, sz, 0, &data);
    std::memcpy(data, kQuadStrip, (size_t)sz);
    vkUnmapMemory(device.device(), quadMem);
}

void UiRenderer::createInstanceBuffer(uint32_t capacity) {
    VkDeviceSize sz = sizeof(RectangleInstance) * capacity;

    device.createBuffer(
        sz,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        instanceVB, instanceMem
    );

    vkMapMemory(device.device(), instanceMem, 0, sz, 0, &instanceMapped);
    currentInstanceCount = 0;
}

void UiRenderer::updateInstances(const std::vector<RectangleInstance>& instances) {
    currentInstanceCount = std::min<uint32_t>((uint32_t)instances.size(), maxInstanceCount);
    if (currentInstanceCount == 0) return;

    std::memcpy(
        instanceMapped,
        instances.data(),
        currentInstanceCount * sizeof(RectangleInstance)
    );
}

void UiRenderer::bind(VkCommandBuffer commandBuffer) {
    VkBuffer bufs[]     = { quadVB, instanceVB };
    VkDeviceSize offs[] = { 0,      0          };
    vkCmdBindVertexBuffers(commandBuffer, /*firstBinding=*/0, /*bindingCount=*/2, bufs, offs);
}

void UiRenderer::draw(VkCommandBuffer commandBuffer) {
    if (currentInstanceCount == 0) return;
    vkCmdDraw(commandBuffer, /*vertexCount=*/4, /*instanceCount=*/currentInstanceCount, /*firstVertex=*/0, /*firstInstance=*/0);
}

std::vector<VkVertexInputBindingDescription> UiRenderer::QuadVertex::getBindDescriptions()
{
    std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(QuadVertex);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescriptions;
}

std::vector<VkVertexInputAttributeDescription> UiRenderer::QuadVertex::getAttributeDescriptions()
{
    std::vector<VkVertexInputAttributeDescription> attributedescriptions(1);
    attributedescriptions[0].binding = 0;
    attributedescriptions[0].location = 0;
    attributedescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributedescriptions[0].offset = offsetof(QuadVertex, position);
    return attributedescriptions;
}

std::vector<VkVertexInputBindingDescription> UiRenderer::RectangleInstance::getBindDescriptions()
{
    std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
    bindingDescriptions[0].binding = 1;
    bindingDescriptions[0].stride = sizeof(RectangleInstance);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    return bindingDescriptions;
}

std::vector<VkVertexInputAttributeDescription> UiRenderer::RectangleInstance::getAttributeDescriptions()
{
    std::vector<VkVertexInputAttributeDescription> attributedescriptions(4);
    attributedescriptions[0].binding = 1;
    attributedescriptions[0].location = 1;
    attributedescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributedescriptions[0].offset = offsetof(RectangleInstance, position);

    attributedescriptions[1].binding = 1;
    attributedescriptions[1].location = 2;
    attributedescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributedescriptions[1].offset = offsetof(RectangleInstance, size);

    attributedescriptions[2].binding = 1;
    attributedescriptions[2].location = 3;
    attributedescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributedescriptions[2].offset = offsetof(RectangleInstance, color);

    attributedescriptions[3].binding = 1;
    attributedescriptions[3].location = 4;
    attributedescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributedescriptions[3].offset = offsetof(RectangleInstance, radius);
    
    return attributedescriptions;
}