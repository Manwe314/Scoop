#include "ShowcaseApp.hpp"
#include <iostream>

// ~~~~~ helpers ~~~~~~

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
  std::cerr << "Showcase validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

static std::vector<uint32_t> uniqueIndices(std::initializer_list<std::optional<uint32_t>> list)
{
    std::vector<uint32_t> out;
    out.reserve(list.size());
    std::unordered_set<uint32_t> seen;
    for (const auto& o : list)
    {
        if (!o.has_value())
            continue;
        uint32_t idx = *o;
        if (seen.insert(idx).second)
            out.push_back(idx);
    }
    return out;
}

VkShaderModule ShowcaseApp::createShaderModule(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create shader module!");
    return module;
}

uint32_t ShowcaseApp::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props, VkPhysicalDevice phys)
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("Showcase: No suitable memory type.");
}

void ShowcaseApp::createBuffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags, VkBuffer& outBuf, VkDeviceMemory& outMem)
{
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size  = size;
    createInfo.usage = usage;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &createInfo, nullptr, &outBuf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, outBuf, &req);

    VkMemoryAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocateInfo.allocationSize  = req.size;
    allocateInfo.memoryTypeIndex = findMemoryType(req.memoryTypeBits, memFlags, phys);

    if (vkAllocateMemory(device, &allocateInfo, nullptr, &outMem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");

    vkBindBufferMemory(device, outBuf, outMem, 0);
}

static VkCommandBuffer beginOneTimeCmd(VkDevice device, VkCommandPool pool)
{
    VkCommandBufferAllocateInfo alocateInfo{};
    alocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alocateInfo.commandPool = pool;
    alocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alocateInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alocateInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

static void endOneTimeCmd(VkDevice device, VkQueue queue, VkCommandPool pool, VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

void copyBuffer(VkDevice device, VkCommandPool pool, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
    VkCommandBuffer cmd = beginOneTimeCmd(device, pool);
    VkBufferCopy region{0,0,size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    endOneTimeCmd(device, queue, pool, cmd);
}

static int getAspectExtent(float aspect, bool isWidth)
{
    if (aspect <= EPSILON)
        return isWidth ? 1200 : 900;

    float width = 1200.0f;
    float height = width / aspect;
    if (height > 1080.0f)
    {
        height = 1080.0f;
        width = aspect * height;
    }
    if (width > 1920.0f)
        return isWidth ? 1920 : static_cast<int>(height);
    return isWidth ? static_cast<int>(width) : static_cast<int>(height);
}

void ShowcaseApp::CharCallback(GLFWwindow*, unsigned int cp)
{
    if (s_active)
        s_active->onChar(cp);
}

void ShowcaseApp::KeyCallback(GLFWwindow*, int key, int, int action, int mods)
{
    if (s_active)
        s_active->onKey(key, action, mods);
}

void ShowcaseApp::onChar(uint32_t cp)
{
    if (cp == 'w')
    {
        scene.camera.position.z -= 0.1;
        scene.camera.target.z -= 0.1;
    }
    if (cp == 's')
    {
        scene.camera.position.z += 0.1;
        scene.camera.target.z += 0.1;
    }
    if (cp == 'a')
    {
        scene.camera.position.x -= 0.1;
        scene.camera.target.x -= 0.1;
    }
    if (cp == 'd')
    {
        scene.camera.position.x += 0.1;
        scene.camera.target.x += 0.1;
    }
    if (cp == 32)
    {
        scene.camera.position.y += 0.1;
        scene.camera.target.y += 0.1;
    }
    
    std::cout << "camera pos " << scene.camera.position.x << " " << scene.camera.position.y << " " << scene.camera.position.z << std::endl;
    
}

void ShowcaseApp::onKey(int key, int action, int )
{

    switch (key) {
        case GLFW_KEY_BACKSPACE:
            break;
        case GLFW_KEY_ENTER:
            break;
        case GLFW_KEY_LEFT_SHIFT:
            scene.camera.position.y -= 0.1;
            scene.camera.target.y -= 0.1;
            break;
        case GLFW_KEY_KP_ENTER:
            break;
        case GLFW_KEY_ESCAPE:
            break;
        default:
            break;
    }
}

void ShowcaseApp::MouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS)
        {
            input.rmbDown = true;
            input.firstDrag = true;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        else if (action == GLFW_RELEASE)
        {
            input.rmbDown = false;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

void ShowcaseApp::CursorPosCallback(GLFWwindow*, double x, double y)
{
    if (!input.rmbDown)
        return;

    if (input.firstDrag)
    {
        input.firstDrag = false;
        input.lastX = x; input.lastY = y;
        return;
    }

    double dx = x - input.lastX;
    double dy = y - input.lastY;
    input.lastX = x; input.lastY = y;

    input.yawDeg   += float(dx) * input.sensitivity;
    input.pitchDeg -= float(dy) * input.sensitivity;

    input.pitchDeg = std::clamp(input.pitchDeg, -89.9f, 89.9f);

    const float cy = std::cos(glm::radians(input.yawDeg));
    const float sy = std::sin(glm::radians(input.yawDeg));
    const float cp = std::cos(glm::radians(input.pitchDeg));
    const float sp = std::sin(glm::radians(input.pitchDeg));

    glm::vec3 forward = glm::normalize(glm::vec3(
        cp * cy,   // x
        sp,        // y
        cp * sy    // z
    ));

    scene.camera.target = scene.camera.position + forward;
}


void ShowcaseApp::initLookAnglesFromCamera()
{
    glm::vec3 f = glm::normalize(scene.camera.target - scene.camera.position);
    input.yawDeg   = glm::degrees(std::atan2(f.z, f.x));
    input.pitchDeg = glm::degrees(std::asin(glm::clamp(f.y, -1.f, 1.f)));
}

// ~~~~ constructor / destructor ~~~~

ShowcaseApp::ShowcaseApp(VkPhysicalDevice gpu, VkInstance inst, Scene scene) : window(getAspectExtent(scene.camera.aspect, true), getAspectExtent(scene.camera.aspect, false), "Scoop"), scene(scene)
{
    instance = inst;
    s_active = this;
    glfwSetMouseButtonCallback(window.handle(), [](GLFWwindow* w, int b, int a, int m){ if (s_active) s_active->MouseButtonCallback(w,b,a,m); });
    glfwSetCursorPosCallback(window.handle(), [](GLFWwindow* w, double x, double y){ if (s_active) s_active->CursorPosCallback(w,x,y); });
    // glfwSetCharCallback(window.handle(), &ShowcaseApp::CharCallback);
    // glfwSetKeyCallback (window.handle(), &ShowcaseApp::KeyCallback);
    setupDebugMessenger();
    window.createWindowSurface(instance, &surface);
    if (gpu == VK_NULL_HANDLE)
        throw std::runtime_error("Showcase: invalid GPU passed");
    physicalDevice = gpu;
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createSyncObjects();
    createOffscreenTargets(); // on resizing will need to call this again
    createComputeDescriptors();
    createComputePipeline();
    createRenderPass();
    createFramebuffers();
    createGraphicsDescriptors();
    createFullscreenGraphicsPipeline();
    createParamsBuffers();
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(gpu, &properties);
    initLookAnglesFromCamera();
    std::cout << "Name: " << properties.deviceName << std::endl;
    DumpScene(ShowcaseApp::scene);
    makeInstances(ShowcaseApp::scene);
}

ShowcaseApp::~ShowcaseApp()
{
    if (graphicsPipeline)
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
    if (graphicsPipelineLayout)
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);

    destroyGraphicsDescriptors();
    destroyFramebuffers();
    
    if (renderPass)
        vkDestroyRenderPass(device, renderPass, nullptr);

    if (computePipeline)
        vkDestroyPipeline(device, computePipeline, nullptr);
    if (computePipelineLayout)
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
    
    destroyComputeDescriptors();
    destroyOffscreenTarget();

    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (paramsMapped[i])
        {
            vkUnmapMemory(device, paramsMemory[i]);
            paramsMapped[i] = nullptr;
        }
        if (paramsBuffer[i]) 
            vkDestroyBuffer(device, paramsBuffer[i], nullptr);
        if (paramsMemory[i])
            vkFreeMemory(device, paramsMemory[i], nullptr);
        if (tlasNodesBuf[i]) vkDestroyBuffer(device, tlasNodesBuf[i], nullptr);
        if (tlasNodesMem[i]) vkFreeMemory(device, tlasNodesMem[i], nullptr);
        if (tlasInstBuf[i])  vkDestroyBuffer(device, tlasInstBuf[i], nullptr);
        if (tlasInstMem[i])  vkFreeMemory(device, tlasInstMem[i], nullptr);
        if (tlasIdxBuf[i])   vkDestroyBuffer(device, tlasIdxBuf[i], nullptr);
        if (tlasIdxMem[i])   vkFreeMemory(device, tlasIdxMem[i], nullptr);
    }
    destorySSBOdata();

    for (auto s : imageRenderFinished)
        vkDestroySemaphore(device, s, nullptr);
    imageRenderFinished.clear();
    
    for (size_t i = 0; i < imageAvailableSemaphores.size(); ++i)
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i)
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    for (size_t i = 0; i < inFlightFences.size(); ++i)
        vkDestroyFence(device, inFlightFences[i], nullptr);
    
    for (auto imageView : swapChainImageViews)
        vkDestroyImageView(device, imageView, nullptr);
    
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    vkDestroyDevice(device, nullptr);
    if (VALIDATE)
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
}

ShowcaseApp* ShowcaseApp::s_active = nullptr;

// ~~~~~ Creation ~~~~~

void ShowcaseApp::makeInstances(Scene& scene)
{
    instances.reserve(scene.objects.size());
    size_t size = scene.meshes.size();
    std::vector<uint32_t> nodeBases(size, 0u);
    std::vector<uint32_t> triBases(size, 0u);
    std::vector<uint32_t> shadeTriBases(size, 0u);
    std::vector<uint32_t> materialBases(size, 0u);
    std::vector<uint32_t> textureBases(size, 0u);

    uint32_t runningNodeSize = 0;
    uint32_t runningTriSize = 0;
    uint32_t runningShaderSize = 0;
    uint32_t runningMatSize = 0;
    uint32_t runningTextureSize = 0;
    int i = 0;
    for (auto& mesh : scene.meshes)
    {
        if (i != 0)
        {
            nodeBases[i] = nodeBases[i - 1] + runningNodeSize;
            triBases[i] = triBases[i - 1] + runningTriSize;
            shadeTriBases[i] = shadeTriBases[i - 1] + runningShaderSize;
            materialBases[i] = materialBases[i - 1] + runningMatSize;
            textureBases[i] = textureBases[i - 1] + runningTextureSize;
        }
        
        runningNodeSize = static_cast<uint32_t>(mesh.bottomLevelAccelerationStructure.nodes.size());
        runningTriSize = static_cast<uint32_t>(mesh.bottomLevelAccelerationStructure.triangles.intersectionTriangles.size());
        runningShaderSize = static_cast<uint32_t>(mesh.bottomLevelAccelerationStructure.triangles.shadingTriangles.size());
        runningMatSize = static_cast<uint32_t>(mesh.perMeshMaterials.size());
        runningTextureSize = static_cast<uint32_t>(mesh.textures.size());
        i++;
    }
    SBVHNodes.reserve(nodeBases[i - 1] + runningNodeSize);
    intersectionTrinagles.reserve(triBases[i - 1] + runningTriSize);
    shadingTriangles.reserve(shadeTriBases[i - 1] + runningShaderSize);
    materials.reserve(materialBases[i - 1] + runningMatSize);
    // textures? .reserve(textureBases[i - 1] + runningTextureSize);
    for (auto& mesh : scene.meshes) {
        auto& sbvh = mesh.bottomLevelAccelerationStructure;

        SBVHNodes.insert(SBVHNodes.end(),
            std::make_move_iterator(sbvh.nodes.begin()),
            std::make_move_iterator(sbvh.nodes.end()));
        sbvh.nodes.clear();

        intersectionTrinagles.insert(intersectionTrinagles.end(),
            std::make_move_iterator(sbvh.triangles.intersectionTriangles.begin()),
            std::make_move_iterator(sbvh.triangles.intersectionTriangles.end()));
        sbvh.triangles.intersectionTriangles.clear();

        shadingTriangles.insert(shadingTriangles.end(),
            std::make_move_iterator(sbvh.triangles.shadingTriangles.begin()),
            std::make_move_iterator(sbvh.triangles.shadingTriangles.end()));
        sbvh.triangles.shadingTriangles.clear();

        materials.insert(materials.end(),
            std::make_move_iterator(mesh.perMeshMaterials.begin()),
            std::make_move_iterator(mesh.perMeshMaterials.end()));
        mesh.perMeshMaterials.clear();
    }

    for (auto& object : scene.objects)
    {
        InstanceData inst{};
        inst.modelToWorld = object.transform.affineTransform();
        inst.worldToModel = affineInverse(inst.modelToWorld);
        inst.worldAABB = object.boundingBox;

        inst.nodeBase = nodeBases[object.meshID];
        inst.triBase = triBases[object.meshID];
        inst.shadeTriBase = shadeTriBases[object.meshID];
        inst.materialBase = materialBases[object.meshID];
        inst.textureBase = textureBases[object.meshID];
        instances.push_back(inst);
    }
}

void ShowcaseApp::createLogicalDevice()
{
    QueueFamiliyIndies indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value(), indices.computeFamily.value()};

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (VALIDATE)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } 
    else
        createInfo.enabledLayerCount = 0;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create logical device!");

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
    graphicsFamily = indices.graphicsFamily.value();
    computeFamily = indices.computeFamily.value();
}

QueueFamiliyIndies ShowcaseApp::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamiliyIndies indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto &queueFamily : queueFamilies)
    {
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphicsFamily = i;

        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
            indices.computeFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (queueFamily.queueCount > 0 && presentSupport)
            indices.presentFamily = i;

        if (indices.isComplete())
            break;
        i++;
    }
    return indices;
}


void ShowcaseApp::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) 
{
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
  createInfo.pUserData = nullptr;
}

void ShowcaseApp::setupDebugMessenger() 
{
    if (!VALIDATE)
        return;
    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("Showcase: failed to set up debug messenger!");
    }
}

SwapChainSupportDetails ShowcaseApp::querrySwapchaindetails(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;
    
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentCount ,nullptr);
    if (presentCount != 0)
    {
        details.presentModes.resize(presentCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentCount ,details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR ShowcaseApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return availableFormat;
    }

    return availableFormats[0];
}

VkPresentModeKHR ShowcaseApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            return availablePresentMode;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D ShowcaseApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;
    else
    {
        VkExtent2D actualExtent = window.getExtent();
        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));   
        return actualExtent;
    }
}

void ShowcaseApp::createSwapchain()
{
    SwapChainSupportDetails swapChainSupport = querrySwapchaindetails(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;

    VkSwapchainKHR oldSwap = validOldSwapchain();

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamiliyIndies indices = findQueueFamilies(physicalDevice);
    std::vector<uint32_t> q = uniqueIndices({indices.graphicsFamily, indices.presentFamily, indices.computeFamily});

    if (q.size() > 1)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = uint32_t(q.size());
        createInfo.pQueueFamilyIndices = q.data();
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwap;

    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &newSwapchain) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create swap chain!");

    if (oldSwap != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(device, oldSwap, nullptr);

    swapChain = newSwapchain;

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void ShowcaseApp::recreateSwapchain()
{
    VkExtent2D extent = window.getExtent();
    while (extent.width == 0 || extent.height == 0) {
        glfwWaitEvents();
        extent = window.getExtent();
    }
    vkDeviceWaitIdle(device);
    destroyFramebuffers();

    for (auto iv : swapChainImageViews)
        vkDestroyImageView(device, iv, nullptr);
    swapChainImageViews.clear();

    for (auto s : imageRenderFinished)
        vkDestroySemaphore(device, s, nullptr);
    imageRenderFinished.clear();

    if (graphicsPipeline)
    {
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (graphicsPipelineLayout)
    {
        vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (renderPass)
    {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }
    destroyOffscreenTarget();

    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createOffscreenTargets();

    updateComputeDescriptor();

    if (graphicsDescPool && graphicsSetLayout)
    {
        for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorImageInfo img{};
            img.sampler     = offscreenSampler;
            img.imageView   = offscreenView[i];
            img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = graphicsSets[i];
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = 1;
            w.pImageInfo = &img;
    
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        }
    }

    createFullscreenGraphicsPipeline();

    imageRenderFinished.resize(swapChainImages.size());
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < imageRenderFinished.size(); ++i)
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageRenderFinished[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create per-image present semaphore!");

    imagesInFlight.assign(swapChainImages.size(), VK_NULL_HANDLE);
    for (int i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
        offscreenInitialized[i] = false;

    window.resetWindowResizedFlag();
}

void ShowcaseApp::createImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.image = swapChainImages[i];
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
           throw std::runtime_error("Showcase: failed to create image views!");
    }

}

void ShowcaseApp::createSyncObjects()
{
    imageAvailableSemaphores.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(SwapChain::MAX_FRAMES_IN_FLIGHT);

    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
    
    imageRenderFinished.resize(swapChainImages.size());

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    
    for (size_t i = 0; i < imageRenderFinished.size(); ++i)
    {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageRenderFinished[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create per-image present semaphore!");
    }

    for (int i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create imageAvailable semaphore!");

        if (vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create renderFinished semaphore!");

        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create inFlight fence!");
    }
}

void ShowcaseApp::createOffscreenTargets()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkExtent3D extent3D{};
        extent3D.width  = swapChainExtent.width;
        extent3D.height = swapChainExtent.height;
        extent3D.depth  = 1;

        VkImageCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        createInfo.imageType = VK_IMAGE_TYPE_2D;
        createInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        createInfo.extent = extent3D;
        createInfo.mipLevels = 1;
        createInfo.arrayLayers = 1;
        createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        createInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &createInfo, nullptr, &offscreenImage[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create offscreen image!");
        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(device, offscreenImage[i], &memReq);
        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            bool typeOk = (memReq.memoryTypeBits & (1u << i)) != 0;
            bool flagsOk = (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            if (typeOk && flagsOk)
            {
                memoryTypeIndex = i;
                break;
            }
        }
        if (memoryTypeIndex == UINT32_MAX)
            throw std::runtime_error("Showcase: no suitable memory type for offscreen image!");
        
        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memReq.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(device, &allocateInfo, nullptr, &offscreenMemory[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to allocate offscreen image memory!");
    
        vkBindImageMemory(device, offscreenImage[i], offscreenMemory[i], 0);

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = offscreenImage[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = createInfo.format;
        viewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;
    
        if (vkCreateImageView(device, &viewCreateInfo, nullptr, &offscreenView[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create offscreen image view!");
        VkSemaphoreCreateInfo semCreateInfo{};
        semCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        if (vkCreateSemaphore(device, &semCreateInfo, nullptr, &computeDone[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create compute semaphore");
    }
}

void ShowcaseApp::createComputeDescriptors()
{
    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b1{};
    b1.binding = 1;
    b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b1.descriptorCount = 1;
    b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b2 = b1;
    b2.binding = 2;

    VkDescriptorSetLayoutBinding b3 = b1;
    b3.binding = 3;

    VkDescriptorSetLayoutBinding b4 = b1;
    b4.binding = 4;

    VkDescriptorSetLayoutBinding b5 = b1;
    b5.binding = 5;

    VkDescriptorSetLayoutBinding b6 = {};
    b6.binding = 6;
    b6.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b6.descriptorCount = 1;
    b6.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b7 = b6;
    b7.binding = 7;

    VkDescriptorSetLayoutBinding b8 = b6;
    b8.binding = 8;

    std::array<VkDescriptorSetLayoutBinding, 9> bindings = {b0, b1, b2, b3, b4, b5, b6, b7, b8};

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = (uint32_t)bindings.size();
    layoutCreateInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &computeSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute descriptor set layout!");

    VkDescriptorPoolSize poolSizeImg{};
    poolSizeImg.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizeImg.descriptorCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
        
    VkDescriptorPoolSize poolSizeBuf{};
    poolSizeBuf.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizeBuf.descriptorCount = 8 * SwapChain::MAX_FRAMES_IN_FLIGHT;
        
    std::array<VkDescriptorPoolSize, 2> poolSizes = {poolSizeImg, poolSizeBuf};

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.maxSets = SwapChain::MAX_FRAMES_IN_FLIGHT;
    poolCreateInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolCreateInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &computeDescPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute descriptor pool!");

    VkDescriptorSetLayout layouts[SwapChain::MAX_FRAMES_IN_FLIGHT];
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
        layouts[i] = computeSetLayout;
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = computeDescPool;
    allocateInfo.descriptorSetCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    allocateInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(device, &allocateInfo, computeSets) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate compute descriptor set!");

    updateComputeDescriptor();
}

void ShowcaseApp::updateComputeDescriptor()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfo.imageView   = offscreenView[i];
        imageInfo.sampler     = VK_NULL_HANDLE;
        
        VkWriteDescriptorSet w0{};
        w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0.dstSet = computeSets[i];
        w0.dstBinding = 0;
        w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0.descriptorCount = 1;
        w0.pImageInfo = &imageInfo;
        
        vkUpdateDescriptorSets(device, 1, &w0, 0, nullptr);
    }
    

}

void ShowcaseApp::writeStaticComputeBindings()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkDescriptorBufferInfo sbvhInfo{ sbvhNodesBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo triangleInfo{ triangleBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadeInfo{ shadingBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo matInfo{ materialBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet w1{};
        w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1.dstSet = computeSets[i];
        w1.dstBinding = 1;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w1.descriptorCount = 1;
        w1.pBufferInfo = &sbvhInfo;

        VkWriteDescriptorSet w2{};
        w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w2.dstSet = computeSets[i];
        w2.dstBinding = 2;
        w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w2.descriptorCount = 1;
        w2.pBufferInfo = &triangleInfo;

        VkWriteDescriptorSet w3{};
        w3.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w3.dstSet = computeSets[i];
        w3.dstBinding = 3;
        w3.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w3.descriptorCount = 1;
        w3.pBufferInfo = &shadeInfo;

        VkWriteDescriptorSet w4{};
        w4.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w4.dstSet = computeSets[i];
        w4.dstBinding = 4;
        w4.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w4.descriptorCount = 1;
        w4.pBufferInfo = &matInfo;

        std::array<VkWriteDescriptorSet, 4> writes = {w1, w2, w3, w4};
        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }
}

void ShowcaseApp::uploadStaticData()
{
    uploadDeviceLocal(SBVHNodes, 0, sbvhNodesBuffer, sbvhNodesMemory);
    uploadDeviceLocal(intersectionTrinagles, 0, triangleBuffer, triangleMemory);
    uploadDeviceLocal(shadingTriangles, 0, shadingBuffer, shadingMemory);
    uploadDeviceLocal(materials, 0, materialBuffer, materialMemory);
    writeStaticComputeBindings();
}

void ShowcaseApp::writeParamsBindingForFrame(uint32_t frameIndex)
{
    VkDescriptorBufferInfo paramsInfo{ paramsBuffer[frameIndex], 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet w5{};
    w5.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w5.dstSet = computeSets[frameIndex];
    w5.dstBinding = 5;
    w5.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w5.descriptorCount = 1;
    w5.pBufferInfo = &paramsInfo;

    vkUpdateDescriptorSets(device, 1, &w5, 0, nullptr);
}

void ShowcaseApp::createParamsBuffers()
{
    for (uint32_t i = 0; i < SwapChain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        createBuffer(device, physicalDevice, sizeof(ParamsGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            paramsBuffer[i], paramsMemory[i]);

        VkResult result = vkMapMemory(device, paramsMemory[i], 0, sizeof(ParamsGPU), 0, &paramsMapped[i]);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Map params buffer failed");
    }
}


void ShowcaseApp::createComputePipeline()
{
    VkPipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = 1;
    layoutCreateInfo.pSetLayouts = &computeSetLayout;
    layoutCreateInfo.pushConstantRangeCount = 0;
    layoutCreateInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &computePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute pipeline layout!");

    auto code = Pipeline::readFile("build/shaders/rayTrace.comp.spv");
    VkShaderModule shader = createShaderModule(code);

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName  = "main";

    VkComputePipelineCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    createInfo.stage = stage;
    createInfo.layout = computePipelineLayout;
    createInfo.basePipelineHandle = VK_NULL_HANDLE;
    createInfo.basePipelineIndex = -1;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &computePipeline) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create compute pipeline!");

    vkDestroyShaderModule(device, shader, nullptr);
}

void ShowcaseApp::createRenderPass()
{
    VkAttachmentDescription color{};
    color.format = swapChainImageFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    createInfo.attachmentCount = 1;
    createInfo.pAttachments = &color;
    createInfo.subpassCount = 1;
    createInfo.pSubpasses = &subpass;
    createInfo.dependencyCount = 1;
    createInfo.pDependencies = &dep;

    if (vkCreateRenderPass(device, &createInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create render pass!");
}

void ShowcaseApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        VkImageView attachments[1] = { swapChainImageViews[i] };

        VkFramebufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        createInfo.renderPass = renderPass;
        createInfo.attachmentCount = 1;
        createInfo.pAttachments = attachments;
        createInfo.width  = swapChainExtent.width;
        createInfo.height = swapChainExtent.height;
        createInfo.layers = 1;

        if (vkCreateFramebuffer(device, &createInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Showcase: failed to create framebuffer!");
    }
}

void ShowcaseApp::createGraphicsDescriptors()
{
    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    if (vkCreateSampler(device, &samplerCreateInfo, nullptr, &offscreenSampler) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create sampler!");

    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = 1;
    layoutCreateInfo.pBindings = &b0;

    if (vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &graphicsSetLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics descriptor set layout!");

    VkDescriptorPoolSize p{};
    p.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    p.descriptorCount = SwapChain::MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.maxSets = SwapChain::MAX_FRAMES_IN_FLIGHT;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &p;

    if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &graphicsDescPool) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics descriptor pool!");

    std::array<VkDescriptorSetLayout, SwapChain::MAX_FRAMES_IN_FLIGHT> layouts{};
    layouts.fill(graphicsSetLayout);
    
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = graphicsDescPool;
    allocateInfo.descriptorSetCount = SwapChain::MAX_FRAMES_IN_FLIGHT;
    allocateInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(device, &allocateInfo, graphicsSets) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to allocate graphics descriptor set!");

    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {

        VkDescriptorImageInfo img{};
        img.sampler = offscreenSampler;
        img.imageView = offscreenView[i];
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = graphicsSets[i];
        w.dstBinding = 0;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.descriptorCount = 1;
        w.pImageInfo = &img;
    
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
}

void ShowcaseApp::createFullscreenGraphicsPipeline()
{
    VkPipelineLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = 1;
    layoutCreateInfo.pSetLayouts = &graphicsSetLayout;

    if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics pipeline layout!");

    auto vertCode = Pipeline::readFile("build/shaders/fullScreen.vert.spv");
    auto fragCode = Pipeline::readFile("build/shaders/fullScreen.frag.spv");

    VkShaderModule vert = createShaderModule(vertCode);
    VkShaderModule frag = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertShaderStageCreateInfo{};
    vertShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageCreateInfo.module = vert;
    vertShaderStageCreateInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageCreateInfo{};
    fragShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageCreateInfo.module = frag;
    fragShaderStageCreateInfo.pName = "main";

    VkPipelineShaderStageCreateInfo stages[2] = { vertShaderStageCreateInfo, fragShaderStageCreateInfo };

    VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
    vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo{};
    inputAssemblyCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width  = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    VkPipelineViewportStateCreateInfo viewportCreateInfo{};
    viewportCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportCreateInfo.viewportCount = 1;
    viewportCreateInfo.pViewports = &viewport;
    viewportCreateInfo.scissorCount = 1;
    viewportCreateInfo.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizationCreateInfo{};
    rasterizationCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationCreateInfo.cullMode = VK_CULL_MODE_NONE;
    rasterizationCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationCreateInfo.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampleCreateInfo{};
    multisampleCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlendCreateInfo{};
    colorBlendCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendCreateInfo.attachmentCount = 1;
    colorBlendCreateInfo.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = stages;
    pipelineCreateInfo.pVertexInputState   = &vertexInputCreateInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyCreateInfo;
    pipelineCreateInfo.pViewportState      = &viewportCreateInfo;
    pipelineCreateInfo.pRasterizationState = &rasterizationCreateInfo;
    pipelineCreateInfo.pMultisampleState   = &multisampleCreateInfo;
    pipelineCreateInfo.pDepthStencilState  = nullptr;
    pipelineCreateInfo.pColorBlendState    = &colorBlendCreateInfo;
    pipelineCreateInfo.layout = graphicsPipelineLayout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("Showcase: failed to create graphics pipeline!");

    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
}


// ~~~~~ Destruction ~~~~~~

void ShowcaseApp::destroyGraphicsDescriptors()
{
    if (graphicsDescPool)
    {
        vkDestroyDescriptorPool(device, graphicsDescPool, nullptr);
        graphicsDescPool = VK_NULL_HANDLE;
    }
    if (graphicsSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, graphicsSetLayout, nullptr);
        graphicsSetLayout = VK_NULL_HANDLE;
    }
    if (offscreenSampler)
    {
        vkDestroySampler(device, offscreenSampler, nullptr);
        offscreenSampler = VK_NULL_HANDLE;
    }
}

void ShowcaseApp::destroyFramebuffers()
{
    for (auto fb : swapChainFramebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    swapChainFramebuffers.clear();
}

void ShowcaseApp::destroyComputeDescriptors()
{
    if (computeDescPool)
    {
        vkDestroyDescriptorPool(device, computeDescPool, nullptr);
        computeDescPool = VK_NULL_HANDLE;
    }
    if (computeSetLayout)
    {
        vkDestroyDescriptorSetLayout(device, computeSetLayout, nullptr);
        computeSetLayout = VK_NULL_HANDLE;
    }
}


void ShowcaseApp::destroyOffscreenTarget()
{
    for (uint32_t i = 0; i < (uint32_t)SwapChain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (computeDone[i])
        {
            vkDestroySemaphore(device, computeDone[i], nullptr);
            computeDone[i] = VK_NULL_HANDLE;
        }
        if (offscreenView[i])
        {
            vkDestroyImageView(device, offscreenView[i], nullptr);
            offscreenView[i] = VK_NULL_HANDLE;
        }
        if (offscreenImage[i])
        {
            vkDestroyImage(device, offscreenImage[i], nullptr);
            offscreenImage[i] = VK_NULL_HANDLE;
        }
        if (offscreenMemory[i])
        {
            vkFreeMemory(device, offscreenMemory[i], nullptr);
            offscreenMemory[i] = VK_NULL_HANDLE;
        }
    }
}

void ShowcaseApp::destorySSBOdata()
{
    if (sbvhNodesBuffer)
        vkDestroyBuffer(device, sbvhNodesBuffer, nullptr);
    if (sbvhNodesMemory)
        vkFreeMemory(device, sbvhNodesMemory, nullptr);

    if (triangleBuffer)
        vkDestroyBuffer(device, triangleBuffer, nullptr);
    if (triangleMemory)
        vkFreeMemory(device, triangleMemory, nullptr);

    if (shadingBuffer)
        vkDestroyBuffer(device, shadingBuffer, nullptr);
    if (shadingMemory)
        vkFreeMemory(device, shadingMemory, nullptr);

    if (materialBuffer)
        vkDestroyBuffer(device, materialBuffer, nullptr);
    if (materialMemory)
        vkFreeMemory(device, materialMemory, nullptr);

}