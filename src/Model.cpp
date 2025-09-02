#include "Model.hpp"



Model::Model(Device& device, const std::vector<Vertex> &vertices) : device(device)
{
    createVertexBuffers(vertices);

}

Model::~Model()
{
    vkDestroyBuffer(device.device(), vertexBuffer, nullptr);
    vkFreeMemory(device.device(), vertexBuffermemory, nullptr);
}
void Model::createVertexBuffers(const std::vector<Vertex> &vertices)
{
    vertexCount = static_cast<uint32_t>(vertices.size());
    assert(vertexCount >= 3 && "Vertex Count must be at least 3");
    VkDeviceSize buffersize = sizeof(vertices[0]) * vertexCount;
    device.createBuffer(buffersize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vertexBuffer, vertexBuffermemory);
    void *data;
    vkMapMemory(device.device(), vertexBuffermemory, 0, buffersize, 0, &data);
    memcpy(data, vertices.data(), static_cast<size_t>(buffersize));
    vkUnmapMemory(device.device(), vertexBuffermemory);
}
void Model::bind(VkCommandBuffer commandBuffer)
{
    VkBuffer buffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);
}
void Model::draw(VkCommandBuffer commandBuffer)
{
    vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
}

std::vector<VkVertexInputBindingDescription> Model::Vertex::getBindDescriptions()
{
    std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(Vertex);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescriptions;
}

std::vector<VkVertexInputAttributeDescription> Model::Vertex::getAttributeDescriptions()
{
    std::vector<VkVertexInputAttributeDescription> attributedescriptions(2);
    attributedescriptions[0].binding = 0;
    attributedescriptions[0].location = 0;
    attributedescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributedescriptions[0].offset = offsetof(Vertex, position);;

    attributedescriptions[1].binding = 0;
    attributedescriptions[1].location = 1;
    attributedescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributedescriptions[1].offset = offsetof(Vertex, color);
    return attributedescriptions;
}