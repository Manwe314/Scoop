#pragma once

#include <string>
#include <vector>
#include "Device.hpp"

typedef struct s_Pipeline
{
    s_Pipeline& operator=(const s_Pipeline&) = delete;

    
    VkRenderPass renderPass = nullptr;
    VkPipelineLayout pipelineLayout = nullptr;

    VkPipelineViewportStateCreateInfo viewportInfo;
    VkPipelineColorBlendStateCreateInfo colorBlendInfo;
    VkPipelineMultisampleStateCreateInfo multisampleInfo;
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
    VkPipelineRasterizationStateCreateInfo rasterizationInfo;
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
    VkPipelineColorBlendAttachmentState colorBlendAttachment;
    VkPipelineDynamicStateCreateInfo dynamicStateInfo;

    std::vector<VkDynamicState> dynamicStateEnables;
    uint32_t subpass = 0;
    
} PipelineConfigInfo ;


class Pipeline
{
private:
    void createGraphicsPipeline(const PipelineConfigInfo& config, const std::string& vertFilepath, const std::string& fragFilepath);
    void createShaderModule(const std::vector<char>& code, VkShaderModule * shaderModule);
    Device& device;
    VkPipeline graphicsPipeline;
    VkShaderModule vertShaderModule;
    VkShaderModule fragShaderModule;
public:
    static std::vector<char> readFile(const std::string& filepath);
    Pipeline(Device& device, const PipelineConfigInfo& config, const std::string& vertFilepath, const std::string& fragFilepath);
    Pipeline(Device& device,
           const PipelineConfigInfo& config,
           const std::vector<VkVertexInputBindingDescription>& bindings,
           const std::vector<VkVertexInputAttributeDescription>& attributes,
           const std::string& vertFilepath,
           const std::string& fragFilepath);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void bind(VkCommandBuffer commandBuffer);

    static void defaultPipelineConfigIngo(PipelineConfigInfo& configInfo);
};

