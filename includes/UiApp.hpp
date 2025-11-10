#pragma once

#include "Window.hpp"
#include "Pipeline.hpp"
#include "clay.h"
#include "Device.hpp"
#include "SwapChain.hpp"
#include "UiRenderer.hpp"
#include "TextRenderer.hpp"
#include "VulkanContext.hpp"
#include "Object.hpp"
#include "SceneUtils.hpp"

#include <memory>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <future>
#include <chrono>
#include <filesystem>
#include <system_error>
#include <nativefiledialog-extended/src/include/nfd.h>

struct AppState {
    bool shouldClose;
    VkPhysicalDevice device;
    Scene scene;
};

struct UiAppState {
    std::string errorMsg;
    std::string selectedDeviceName;
    bool showError = false;
    bool showDevicePicker = false;
    bool isReadyToDisplay = false;
    bool loadingModel = false;
    bool hasObject = false;
    int targetEditor = -1;
    int previousTargetEditor = -1;
    std::vector<bool> isObjectReady;
};

struct TextInputState {
    std::string text;
    bool focused = false;

    int    fontPx = 32;
    float  letterSpacing = 1.0f;
    size_t caret = 0;
    double blinkStart = 0.0;
};

static inline uint64_t keyFrom(Clay_ElementId id) { return (uint64_t)id.id; }

struct TextInputStore {
    std::unordered_map<uint64_t, TextInputState> fields;
    Clay_ElementId focusedId {0};
    Clay_BoundingBox focusedRect{};

    TextInputState& get(Clay_ElementId id) { return fields[keyFrom(id)]; }
    
    TextInputState* focused()
    {
        if (focusedId.id == 0) return nullptr;
        auto it = fields.find(keyFrom(focusedId));
        return (it==fields.end()) ? nullptr : &it->second;
    }

    void focus(Clay_ElementId id)
    {
        if (focusedId.id) fields[keyFrom(focusedId)].focused = false;
        focusedId = id;
        auto& s = fields[keyFrom(id)];
        s.focused = true;
        s.caret   = s.text.size();
        s.blinkStart = glfwGetTime();
    }
    
    void blurAll()
    {
        if (focusedId.id) fields[keyFrom(focusedId)].focused = false;
        focusedId = Clay_ElementId{0};
    }
};

struct TextRun {
    int px;
    VkRect2D scissor;
    uint32_t first = 0;
    uint32_t count = 0;
    int zIndex = 0;
    uint32_t seq = 0;
};

struct RunTemp {
    int px;
    VkRect2D scissor;
    std::vector<TextRenderer::GlyphInstance> glyphs;
    int z;
    uint32_t seq;
};

using Sid = uint32_t;

struct ModelTab {
    std::string name;
    std::shared_ptr<Object> object;
    Sid stableID;
};

struct PendingBVH {
    std::string name;
    size_t      modelIdx;
    std::future<ObjectMeshData> fut;
};


class UiApp
{
    public:
        static const int WIDTH = 1200;
        static const int HEIGHT = 800;
        UiApp(std::string def, VulkanContext& context);
        ~UiApp();
        AppState run();
        UiApp(const UiApp&) = delete;
        void operator=(const UiApp&) = delete;
        
        void HandleButtonInteraction(Clay_ElementId elementId, Clay_PointerData pointerData);
        void HandleErrorShowing(Clay_ElementId elementId, Clay_PointerData pointerData);
        void HandleFloatingShowing(Clay_ElementId elementId, Clay_PointerData pointerData);
        
        
        static void hoverBridge(Clay_ElementId id, Clay_PointerData pd, intptr_t user) {
            auto* self = reinterpret_cast<UiApp*>(user);
            if (self)
            {
                if (self->uiState.showError)
                    self->HandleErrorShowing(id, pd);
                else if (self->uiState.showDevicePicker)
                    self->HandleFloatingShowing(id, pd);
                else
                    self->HandleButtonInteraction(id, pd);
            }
        }
        void resetUiApp();
        
        
    private:
        void loadUi();
        void createPipelineLayout();
        void createTextPipelineLayout();
        void createPipeline();
        void createTextPipeline();
        void createCommandBuffers();
        void freeCommandBuffers();
        void drawFrame();
        void recreateSwapchain();
        void recordCommandBuffer(int imageIndex);
        void reconstruct();
        
        static UiApp* s_active;
        static void CharCallback(GLFWwindow* win, unsigned int codepoint);
        static void KeyCallback (GLFWwindow* win, int key, int scancode, int action, int mods);
        
        void onChar(uint32_t cp);
        void onKey (int key, int action, int mods);
        void HandleMultiInput(Clay_ElementId elementId, Clay_PointerData pointerData);
        bool isInput(Clay_ElementId elementId);
        bool isModelButton(Clay_ElementId elementId);
        void UpdateInput(bool sameIndex = false);
        void pollBVHBuilders();
        void finalizeSceneForLaunch();
        
        TextInputStore inputs;
        
        
        Window window;
        Device device;
        
        AppState state;
        UiAppState uiState;
        bool exitRun = false;
        
        GLFWcursor* cursorArrow = nullptr;
        GLFWcursor* cursorHand  = nullptr;
        GLFWcursor* cursorIBeam = nullptr;
        
        std::vector<std::string> clayFrameStrings;
        
        std::unique_ptr<SwapChain> swapChain;
        
        std::unique_ptr<Pipeline> textPipeline;
        std::unique_ptr<Pipeline> pipeline;
        VkPipelineLayout textPipelineLayout;
        VkPipelineLayout pipelineLayout;
        
        std::vector<VkCommandBuffer> commandBuffers;
        
        std::unique_ptr<UiRenderer> ui;
        std::unique_ptr<UiRenderer> uiOverlay;
        std::unique_ptr<TextRenderer> text;
        
        std::vector<UiRenderer::RectangleInstance> rects;
        std::unordered_map<int, std::vector<TextRenderer::GlyphInstance>> textBatches;
        
        GLFWcursor* wanted;
        
        std::vector<TextRun> textRuns;
        std::vector<RunTemp> tempRuns;
        Clay_ElementId focusedInputId{0};
        Clay_BoundingBox focusedInputRect{0,0,0,0};
        
        uint64_t clayMemSize = 0;
        void*    clayMem     = nullptr;
        Clay_Arena clayArena{};
        
        Sid nextSid = 1;
        std::vector<ModelTab> models;
        std::unordered_map<Sid, size_t> sidToIndex;
        std::future<Object> fut;
        

        std::vector<PendingBVH>         pendingBVH;
        std::unordered_set<std::string> bvhInFlight; 
        
        std::vector<VkPhysicalDevice> optionalDevices;
        std::vector<std::string> optionalDeviceNames;
        
        std::vector<Clay_ElementId> inputFields = {
            Clay_GetElementId(CLAY_STRING("cam.position.input.x")),
            Clay_GetElementId(CLAY_STRING("cam.position.input.y")),
            Clay_GetElementId(CLAY_STRING("cam.position.input.z")),
            Clay_GetElementId(CLAY_STRING("cam.target.input.x")),
            Clay_GetElementId(CLAY_STRING("cam.target.input.y")),
            Clay_GetElementId(CLAY_STRING("cam.target.input.z")),
            Clay_GetElementId(CLAY_STRING("cam.up.input.x")),
            Clay_GetElementId(CLAY_STRING("cam.up.input.y")),
            Clay_GetElementId(CLAY_STRING("cam.up.input.z")),
            Clay_GetElementId(CLAY_STRING("cam.vfov.input")),
            Clay_GetElementId(CLAY_STRING("cam.aspect.input")),
            Clay_GetElementId(CLAY_STRING("cam.near.input")),
            Clay_GetElementId(CLAY_STRING("cam.far.input")),
            Clay_GetElementId(CLAY_STRING("input form")),
        };

        inline void rebuildSidIndexMap()
        {
            sidToIndex.clear();
            for (size_t i = 0; i < models.size(); ++i)
                sidToIndex[models[i].stableID] = i;
        }

        static inline std::optional<Sid> sidForIndex(const std::vector<ModelTab>& models, int idx)
        {
            if (idx < 0 || idx >= (int)models.size())
                return std::nullopt;
            return models[idx].stableID;
        }

        inline bool allBVHReady()
        {
            for (auto& m : models)
            {
                bool isBuilt = false;
                for (auto& s : state.scene.meshes)
                {
                    if (s.name == m.name)
                    {
                        isBuilt = true;
                        break;
                    }
                }
                if (!isBuilt)
                    return false;
            }
            if (models.size() < 1)
                return false;
            return true;
        }
        
        void buildUi();
        // void renderUi(const Clay_RenderCommandArray& cmds, int screenW, int screenH);
    };
    
    
    