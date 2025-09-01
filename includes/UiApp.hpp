#pragma once

#include "Window.hpp"
#include "Pipeline.hpp"
#include "clay.h"
#include "Device.hpp"
#include "SwapChain.hpp"

#include <memory>
#include <vector>
#include <stdexcept>

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
        void createPipelineLayout();
        void createPipeline();
        void createCommandBuffers();
        void drawFrame();

        Window window;
        Device device;
        SwapChain swapChain;
        std::unique_ptr<Pipeline> pipeline;
        VkPipelineLayout pipelineLayout;
        std::vector<VkCommandBuffer> commandBuffers;

        // uint64_t clayMemSize = 0;
        // void*    clayMem     = nullptr;
        // Clay_Arena clayArena{};

        // void buildUi(float dt, int screenW, int screenH);
        // void renderUi(const Clay_RenderCommandArray& cmds, int screenW, int screenH);
};


