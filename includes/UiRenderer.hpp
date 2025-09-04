#pragma once

#include "Device.hpp"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <cassert>
#include <cstring>


class UiRenderer
{
    public:
    
        struct QuadVertex
        {
            glm::vec2 position;
            static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
            static std::vector<VkVertexInputBindingDescription> getBindDescriptions();
        };

        struct RectangleInstance
        {
            glm::vec2 position;
            glm::vec2 size;
            glm::vec4 color;
            glm::vec4 radius; 
            static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
            static std::vector<VkVertexInputBindingDescription> getBindDescriptions();
        };


        UiRenderer(Device& device, uint32_t maxInstances);
        ~UiRenderer();



        UiRenderer(const UiRenderer&) = delete;
        UiRenderer& operator=(const UiRenderer&) = delete;


        void updateInstances(const std::vector<RectangleInstance>& instances);
        void bind(VkCommandBuffer commandBuffer);
        void draw(VkCommandBuffer commandBuffer);

        uint32_t instanceCount() const { return currentInstanceCount; }
        uint32_t instanceCapacity() const { return maxInstanceCount; }


    private:
        void*          instanceMapped = nullptr;
        uint32_t       maxInstanceCount = 0;
        uint32_t       currentInstanceCount = 0;

        Device& device;

        VkBuffer       quadVB = VK_NULL_HANDLE;
        VkDeviceMemory quadMem = VK_NULL_HANDLE;

        VkBuffer       instanceVB = VK_NULL_HANDLE;
        VkDeviceMemory instanceMem = VK_NULL_HANDLE;

        void createQuadBuffer();
        void createInstanceBuffer(uint32_t capacity);
};


