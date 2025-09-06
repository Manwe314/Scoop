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

class UiApp
{
    public:
        static const int WIDTH = 800;
        static const int HEIGHT = 600;
        UiApp();
        ~UiApp();
        void run();
        UiApp(const UiApp&) = delete;
        void operator=(const UiApp&) = delete;
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

        uint64_t clayMemSize = 0;
        void*    clayMem     = nullptr;
        Clay_Arena clayArena{};

        void buildUi();
        // void renderUi(const Clay_RenderCommandArray& cmds, int screenW, int screenH);
};


