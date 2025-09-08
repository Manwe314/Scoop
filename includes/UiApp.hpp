#pragma once

#include "Window.hpp"
#include "Pipeline.hpp"
#include "clay.h"
#include "Device.hpp"
#include "SwapChain.hpp"
#include "UiRenderer.hpp"
#include "TextRenderer.hpp"

#include <memory>
#include <vector>
#include <stdexcept>
#include <unordered_map>

struct TextInputState {
    std::string text;
    bool focused = false;
};

class UiApp
{
    public:
        static const int WIDTH = 1200;
        static const int HEIGHT = 800;
        UiApp();
        ~UiApp();
        void run();
        UiApp(const UiApp&) = delete;
        void operator=(const UiApp&) = delete;
        
        void HandleButtonInteraction(Clay_ElementId elementId, Clay_PointerData pointerData);
        
        
        static void hoverBridge(Clay_ElementId id, Clay_PointerData pd, intptr_t user) { auto* self = reinterpret_cast<UiApp*>(user); if (self) self->HandleButtonInteraction(id, pd); }
        
        
        TextInputState getInputState() { return input; }
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
        
        static UiApp* s_active;
        static void CharCallback(GLFWwindow* win, unsigned int codepoint);
        static void KeyCallback (GLFWwindow* win, int key, int scancode, int action, int mods);

        void onChar(uint32_t cp);
        void onKey (int key, int action, int mods);
        
        
        TextInputState input;
        
        Window window;
        Device device;
        
        std::unique_ptr<SwapChain> swapChain;
        
        std::unique_ptr<Pipeline> textPipeline;
        std::unique_ptr<Pipeline> pipeline;
        VkPipelineLayout textPipelineLayout;
        VkPipelineLayout pipelineLayout;
        
        std::vector<VkCommandBuffer> commandBuffers;

        std::unique_ptr<UiRenderer> ui;
        std::unique_ptr<TextRenderer> text;

        std::vector<UiRenderer::RectangleInstance> rects;
        std::unordered_map<int, std::vector<TextRenderer::GlyphInstance>> textBatches;

        GLFWcursor* wanted = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

        uint64_t clayMemSize = 0;
        void*    clayMem     = nullptr;
        Clay_Arena clayArena{};

        void buildUi();
        // void renderUi(const Clay_RenderCommandArray& cmds, int screenW, int screenH);
};


