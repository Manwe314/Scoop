#pragma once

#include "Window.hpp"
#include "Pipeline.hpp"
#include "clay.h"
#include "Device.hpp"
#include "SwapChain.hpp"
#include "Object.hpp"
#include "SceneUtils.hpp"
#include "Utils.hpp"
#include "WavefrontHelpers.hpp"

#include <optional>
#include <set>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <initializer_list>



#define VALIDATE true
#define FPS true
static constexpr bool SimpleRayTrace = false;

struct GpuTexture {
    VkImage        image        = VK_NULL_HANDLE;
    VkImageView    view         = VK_NULL_HANDLE;
    VkDeviceMemory memory       = VK_NULL_HANDLE;
    uint32_t       width        = 0;
    uint32_t       height       = 0;
    uint32_t       mipLevels    = 1;
    VkFormat       format       = VK_FORMAT_UNDEFINED;
};

struct EmisiveTriangle {
    uint primitiveIndex;
    uint instanceIndex;
    uint materialIndex;
    uint padding;
};


struct FrameUpload {
    VkBuffer        staging = VK_NULL_HANDLE;
    VkDeviceMemory  stagingMem = VK_NULL_HANDLE;
    void*           mapped = nullptr;
    VkDeviceSize    capacity = 0;

    VkCommandPool   uploadPool = VK_NULL_HANDLE;
    VkCommandBuffer uploadCB = VK_NULL_HANDLE;

    VkSemaphore     uploadDone = VK_NULL_HANDLE;
    VkFence         uploadFence = VK_NULL_HANDLE;

};

struct InputState {
    bool rmbDown = false;
    bool firstDrag = true;
    double lastX = 0.0;
    double lastY = 0.0;
    float yawDeg   = 0.0f;
    float pitchDeg = 0.0f;
    float sensitivity = 0.1f;
};

struct QueueFamiliyIndies
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> transferFamily;

    bool hasDedicatedTransfer   = false;
    bool hasSeparateTransfer    = false;
    bool canSplitComputeXfer    = false;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value(); }
};

struct alignas(16) ParamsGPU {
    Mat4 viewProjInv;
    glm::vec4 camPos_time;
    glm::uvec2 imageSize;
    uint32_t   rootIndex;
    uint32_t   flags;
};

void copyBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size);

struct InstanceData {
    AffineMatrix modelToWorld;
    AffineMatrix worldToModel;

    AABB worldAABB;

    uint32_t nodeBase = 0;
    uint32_t triBase = 0;
    uint32_t shadeTriBase = 0;
    uint32_t shadeTriCount = 0;
    uint32_t materialBase = 0;
    uint32_t textureBase = 0;
    uint32_t lightMapBase = 0;
};

struct TLASNode {
    AABB bounds;
    uint32_t first = 0;
    uint32_t count = 0;

    int32_t left  = -1;
    int32_t right = -1;

    bool isLeaf() const { return count > 0; }
};

struct TLAS {
    std::vector<TLASNode>   nodes;
    std::vector<uint32_t>   instanceIndices;
    std::vector<InstanceData> instances;
    uint32_t root;

    void clear() {
        nodes.clear();
        instanceIndices.clear();
        instances.clear();
    }
};

struct alignas(16) TLASNodeGPU {
    glm::vec4 bmin; //w unused
    glm::vec4 bmax; //w unused
    glm::uvec4 meta; // first & count | left & right
};

struct alignas(16) InstanceDataGPU {
    glm::vec4 modelToWorld[3];
    glm::vec4 worldToModel[3];
    glm::vec4 aabbMin; // w unused
    glm::vec4 aabbMax; // w unused
    glm::uvec4 bases0; // nodeBase, triBase, shadeTriBase, materialBase
    glm::uvec4 bases1; // textureBase, sbvhRoot, flags, lightBase
};

class ShowcaseApp
{
private:
    VkInstance instance;
    Window window;
    VkDevice device;

    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    std::vector<VkImageView> swapChainImageViews;
    bool offscreenValid[SwapChain::MAX_FRAMES_IN_FLIGHT] = {false};


    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;
    VkQueue transferQueue;

    VkQueryPool queryPool = VK_NULL_HANDLE;
    float timestampPeriodNs = 0.0f;
    bool queryPrimed[SwapChain::MAX_FRAMES_IN_FLIGHT]{false};

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> imageRenderFinished;
    std::vector<VkFence>     inFlightFences;
    std::vector<VkFence>     imagesInFlight;
    VkFence imageAcquiredFences[SwapChain::MAX_FRAMES_IN_FLIGHT]{};


    uint32_t currentFrame = 0;
    VkSampler textureSampler = VK_NULL_HANDLE;

    InputState input;

    VkImage        offscreenImage[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory offscreenMemory[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkImageView    offscreenView[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkDescriptorSetLayout computeStaticSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeFrameSetLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDynamicSetLayout = VK_NULL_HANDLE;

    VkDescriptorPool      computeDescPool         = VK_NULL_HANDLE;

    VkDescriptorSet       computeStaticSet        = VK_NULL_HANDLE;
    VkDescriptorSet       computeFrameSets[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDescriptorSet       computeDynamicSets[SwapChain::MAX_FRAMES_IN_FLIGHT]{}; 

    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;

    VkPipeline       computePipeline       = VK_NULL_HANDLE;

    VkPipeline rayTraceLogicPipeline     = VK_NULL_HANDLE;
    VkPipeline rayTraceNewPathPipeline   = VK_NULL_HANDLE;
    VkPipeline rayTraceMaterialPipeline  = VK_NULL_HANDLE;
    VkPipeline rayTraceExtendRayPipeline = VK_NULL_HANDLE;
    VkPipeline rayTraceShadowRayPipeline = VK_NULL_HANDLE;

    VkRenderPass          renderPass = VK_NULL_HANDLE;

    VkSampler             offscreenSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout graphicsSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      graphicsDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet       graphicsSets[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkPipelineLayout      graphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            graphicsPipeline       = VK_NULL_HANDLE;

    VkSemaphore computeDone[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer sbvhNodesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory sbvhNodesMemory = VK_NULL_HANDLE;

    VkBuffer triangleBuffer = VK_NULL_HANDLE;
    VkDeviceMemory triangleMemory = VK_NULL_HANDLE;

    VkBuffer shadingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory shadingMemory = VK_NULL_HANDLE;

    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceMemory materialMemory = VK_NULL_HANDLE;

    VkBuffer emissiveTriangleBuffer = VK_NULL_HANDLE;
    VkDeviceMemory emissiveTriangleMemory = VK_NULL_HANDLE;

    VkBuffer lightProbBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lightProbMemory = VK_NULL_HANDLE;

    VkBuffer lightPdfBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lightPdfMemory = VK_NULL_HANDLE;

    VkBuffer lightAliasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lightAliasMemory = VK_NULL_HANDLE;

    VkBuffer triToLightIdxBuffer = VK_NULL_HANDLE;
    VkDeviceMemory triToLightIdxMemory = VK_NULL_HANDLE;

    VkBuffer       pathHeaderBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory pathHeaderMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       rayBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory rayMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       hitIdsBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory hitIdsMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       hitDataBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory hitDataMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       radianceBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory radianceMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       bsdfSampleBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory bsdfSampleMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       lightSampleBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory lightSampleMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       shadowRayBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory shadowRayMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       shadowResultBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory shadowResultMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       pathQueueBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory pathQueueMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer      pixelStatsBuf   [SwapChain::MAX_FRAMES_IN_FLIGHT] = {};
    VkDeviceMemory pixelStatsMem  [SwapChain::MAX_FRAMES_IN_FLIGHT] = {};

    VkBuffer      highVarPixelBuf [SwapChain::MAX_FRAMES_IN_FLIGHT] = {};
    VkDeviceMemory highVarPixelMem[SwapChain::MAX_FRAMES_IN_FLIGHT] = {};

    VkBuffer      adaptiveCountersBuf   [SwapChain::MAX_FRAMES_IN_FLIGHT] = {};
    VkDeviceMemory adaptiveCountersMem  [SwapChain::MAX_FRAMES_IN_FLIGHT] = {};


    VkBuffer       paramsBuffer[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory paramsMemory[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    void*          paramsMapped[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       tlasNodesBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory tlasNodesMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       tlasInstBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory tlasInstMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkBuffer       tlasIdxBuf[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkDeviceMemory tlasIdxMem[SwapChain::MAX_FRAMES_IN_FLIGHT]{};

    VkCommandPool graphicsCommandPool = VK_NULL_HANDLE;
    VkCommandPool computeCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer graphicsCommandBuffers[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    VkCommandBuffer computeCommandBuffers[SwapChain::MAX_FRAMES_IN_FLIGHT]{};
    bool offscreenInitialized[SwapChain::MAX_FRAMES_IN_FLIGHT] = {false};

    uint32_t graphicsFamily = VK_QUEUE_FAMILY_IGNORED;
    uint32_t computeFamily  = VK_QUEUE_FAMILY_IGNORED;
    uint32_t presentFamily = VK_QUEUE_FAMILY_IGNORED;
    uint32_t transferFamily = VK_QUEUE_FAMILY_IGNORED;

    bool hasTransferQueue = false;
    bool hasDedicatedTransfer = false;
    
    void imageBarrier(VkCommandBuffer cmd, VkImage image,
                            VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                            VkPipelineStageFlags dstStage, VkAccessFlags dstAccess,
                            VkImageLayout oldLayout, VkImageLayout newLayout,
                            uint32_t srcQue, uint32_t dstQue);
    
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSurfaceKHR surface;
    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
    const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME};

    FrameUpload frameUpload[SwapChain::MAX_FRAMES_IN_FLIGHT];
    VkFence computeFences[SwapChain::MAX_FRAMES_IN_FLIGHT]{};


    Scene scene;
    TLAS topLevelAS;
    std::vector<InstanceData> instances;
    std::vector<SBVHNode> SBVHNodes;
    std::vector<MollerTriangle> intersectionTrinagles;
    std::vector<ShadingTriangle> shadingTriangles;
    std::vector<EmisiveTriangle> emissiveTrinagles;
    std::vector<float> lightPdf;
    std::vector<float> lightProb;
    std::vector<uint32_t> lightAlias;
    std::vector<MaterialGPU> materials;
    std::map<std::string, uint32_t> textureIndexMap;
    std::vector<ImageRGBA8> flattened;
    std::vector<GpuTexture> gpuTextures;
    std::vector<uint32_t> triToLightIdx;
    

    static ShowcaseApp* s_active;

    static void CharCallback(GLFWwindow* win, unsigned int codepoint);
    static void KeyCallback (GLFWwindow* win, int key, int scancode, int action, int mods);
    
    void onChar(uint32_t cp);
    void onKey (int key, int action, int mods);
    bool viewFaces = false;
    int selectedInstance = -1;
    bool bTransitionActive = false;
    float bInterpTime = 0.0f;
    float bTransitionDuration = 1.5f;
    uint32_t bInterpInt = 0;

    
    void makeInstances(Scene& scene);
    void update(float dt);
    void initLookAnglesFromCamera();
    void initUploadResources();
    void ensureUploadStagingCapacity(uint32_t frameIndex, VkDeviceSize need);
    void MouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
    void CursorPosCallback(GLFWwindow*, double x, double y);
    std::vector<const char *> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    void ensureBufferCapacity(VkBuffer& buf, VkDeviceMemory& mem, VkDeviceSize neededSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags);
    void uploadTLASForFrame(uint32_t frameIndex, const std::vector<TLASNodeGPU>& tlasNodes, const std::vector<InstanceDataGPU>& tlasInstances, const std::vector<uint32_t>& instanceIndices);
    void frameTLASPrepare(uint32_t frameIndex);
    void setupDebugMessenger();
    QueueFamiliyIndies findQueueFamilies(VkPhysicalDevice device);
    void makeEmissionTriangles();
    void createOrResizeWavefrontBuffers();
    void createLogicalDevice();
    void createSwapchain();
    void recreateSwapchain();
    void createImageViews();
    void createSyncObjects();
    void createOffscreenTargets();
    void destroyOffscreenTarget();
    void createComputeDescriptors();
    void updateComputeDescriptor();
    void destroyComputeDescriptors();
    void createComputePipeline();
    void createRenderPass();
    void createFramebuffers();
    void destroyFramebuffers();
    void createGraphicsDescriptors();
    void destroyGraphicsDescriptors();
    void createFullscreenGraphicsPipeline();
    void createCommandPoolAndBuffers();
    void createQueryPool();
    void destroyCommandPoolAndBuffers();
    void destroySceneTextures();
    void destroySSBOdata();
    void uploadStaticData();
    void createParamsBuffers();
    void writeStaticComputeBindings();
    void writeParamsBindingForFrame(uint32_t frameIndex);
    void createTextureSampler(VkPhysicalDevice phys, float requestedAniso = 8.0f);
    void recordComputeCommands(uint32_t i);
    void recordGraphicsCommands(uint32_t frameIndex, uint32_t swapImageIndex);
    void createBuffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags, VkBuffer& outBuf, VkDeviceMemory& outMem);
    void uploadTextureImages();
    void createWavefrontBuffers();
    void destroyWavefrontBuffers();
    uint32_t getMaxPaths() const;
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props, VkPhysicalDevice phys);
    SwapChainSupportDetails querrySwapchaindetails(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void FlattenSceneTexturesAndRemapMaterials();
    GpuTexture createTextureFromImageRGBA8(const ImageRGBA8& img, VkDevice device, VkPhysicalDevice phys, VkFormat format, bool generateMips);
    
    inline VkSwapchainKHR validOldSwapchain() const { return swapChain != VK_NULL_HANDLE ? swapChain : VK_NULL_HANDLE; }

    inline ParamsGPU makeDefaultParams(uint32_t width = 1, uint32_t height = 1, uint32_t rootIndex = 0, float time = 0.0f, const glm::vec3& camPos = glm::vec3(0.0f))
    {
        ParamsGPU p{};
        p.viewProjInv = Mat4(1.0f);
        p.camPos_time = glm::vec4(camPos, time);
        p.imageSize   = glm::uvec2(width, height);
        p.rootIndex   = rootIndex;
        p.flags       = 0;
        return p;
    }

    inline ParamsGPU makeDefaultParams(VkExtent2D extent, uint32_t rootIndex = 0, float time = 0.0f, const glm::vec3& camPos = glm::vec3(0.0f)) { return makeDefaultParams(extent.width, extent.height, rootIndex, time, camPos); }

public:
    ShowcaseApp(VkPhysicalDevice gpu, VkInstance inst, Scene scene);
    ~ShowcaseApp();
    void run();
    template<class T>
    void uploadDeviceLocal(const std::vector<T>& src, VkBufferUsageFlags extraUsage, VkBuffer& outBuf, VkDeviceMemory& outMem)
    {
        VkDeviceSize size = sizeof(T) * src.size();

        VkBuffer stageBuf = VK_NULL_HANDLE;
        VkDeviceMemory stageMem = VK_NULL_HANDLE;
        this->createBuffer(device, physicalDevice, size,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stageBuf, stageMem);

        void* p = nullptr;
        vkMapMemory(device, stageMem, 0, size, 0, &p);
        std::memcpy(p, src.data(), size);
        vkUnmapMemory(device, stageMem);

        this->createBuffer(device, physicalDevice, size,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outBuf, outMem);

        copyBuffer(device, computeCommandPool, computeQueue, stageBuf, outBuf, size);

        vkDestroyBuffer(device, stageBuf, nullptr);
        vkFreeMemory(device, stageMem, nullptr);
    }
};
