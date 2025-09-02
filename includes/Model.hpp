#pragma once

#include "Device.hpp"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <cassert>
#include <cstring>


class Model
{
    public:
    
    struct Vertex
    {
        glm::vec2 position;
        glm::vec3 color;
        static std::vector<VkVertexInputBindingDescription> getBindDescriptions();
        static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
    };


    Model(Device& device, const std::vector<Vertex> &vertices);
    ~Model();
    
    
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    
    
    void bind(VkCommandBuffer commandBuffer);
    void draw(VkCommandBuffer commandBuffer);
private:
    Device& device;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBuffermemory;
    uint32_t vertexCount;
    void createVertexBuffers(const std::vector<Vertex> &vertices);

};


