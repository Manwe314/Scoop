#pragma once

#include "Window.hpp"
#include "Pipeline.hpp"
#include "clay.h"
#include "Device.hpp"
#include "SwapChain.hpp"


class ShowcaseApp
{
private:
    Device device;
public:
    ShowcaseApp(Device device);
    ~ShowcaseApp();
};