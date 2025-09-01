#pragma once

#include <string>
#include <vector>
#include "Device.hpp"

typedef struct s_Pipeline
{
    VkRect2D scissor;
    VkViewport viewport;
    VkRenderPass renderPass = nullptr;
    VkPipelineLayout pipelineLayout = nullptr;
    VkPipelineColorBlendStateCreateInfo colorBlendInfo;
    VkPipelineMultisampleStateCreateInfo multisampleInfo;
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
    VkPipelineRasterizationStateCreateInfo rasterizationInfo;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    uint32_t subpass = 0;
    
} PipelineConfigInfo ;


class Pipeline
{
private:
    static std::vector<char> readFile(const std::string& filepath);
    void createGraphicsPipeline(const PipelineConfigInfo& config, const std::string& vertFilepath, const std::string& fragFilepath);
    void createShaderModule(const std::vector<char>& code, VkShaderModule * shaderModule);
    Device& device;
    VkPipeline graphicsPipeline;
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
public:
    Pipeline(Device& device, const PipelineConfigInfo& config, const std::string& vertFilepath, const std::string& fragFilepath);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    void operator=(const Pipeline&) = delete;

    void bind(VkCommandBuffer commandBuffer);

    static PipelineConfigInfo defaultPipelineConfigIngo(uint32_t width, uint32_t height);
};

