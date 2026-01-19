# Scoop

===

This is a school 42 project about parsing and rendering a 3D obj file in 3D space. 
As the scope of the project dictates there is a free choice of programing language and graphics API
I went with **C++** and **Vulkan** writing shaders in **GLSL**
The Project also does not limit us in the rendering methodology although clasic raseterization is recomended
Netherless I went with RayCasting and Full PathTracing as my rendering strategy.
I also added a GUI app layer to the program to help set up scenes.

## Table of Contents

===

- [Project Overview](#project-overview)
    - [Introduction](#introduction)
    - [Dependencies and Technologies](#Dependencies-and-Technologies)
- [Important Note](#important-note)
- [Warning](#warning)
- [Dependencies and Prerequisites](#dependencies-and-prerequisites)
- [How to Build and Run](#how-to-build-and-run)
- [Basic Usage of the Application](#basic-usage-of-the-application)
- [Configuration](#configuration)
- [Controls](#controls)
- [Theory of Raytracing](#theory-of-raytracing)
- [Assets and Shaders](#assets-and-shaders)
- [Notes about the Code and Optimizations](#notes-about-the-code-and-optimizations)
- [Testing and Validation](#testing-and-validation)
- [Performance and Benchmarking](#performance-and-benchmarking)
- [Possible Future Upgrades](#possible-future-upgrades)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

===

### Introduction

---

The minimal requierement of the project is to have the program take as input the path to an `.obj` file.
Program parses it and then displays in 3D the object rotating along its center axis. 
Additionally users must be able to move the object in space along the 6 cardinal directions and with a dedicated button apply a texture thats defined in the `.mtl` file for that object.

However I personally elevated the project by adding first a GUI app layer where users can add `.obj` files and also
set up the scene by giving each object a transform (position, rotation and scale). aditionall features were added to also alter the camera along with the ability to move the camera using `WASD` while rendering the scene.

Additionally I've added in [ShowcaseApp.hpp](/includes/ShowcaseApp.hpp) `static constexpr bool SimpleRayTrace = true;`
By default set to `true` `SimlerayTrace` boolean compiles 2 versions of the application. if true when compiled the application will use simple **Ray Casting** to render the 3D models. this version of the app was developed to closely resemble the application as asked for by the school project. if `SimplerayTrace` is set to `false` the application will use the full **Real time Phisically based Path Tracing** 


### Dependencies and Technologies
---

Scoop was developed using C++20. In escence the application should be fully cross platform and runable from both windows, mac and linux but the build system in place is only for linux.

The requirements are:

- [Vulkan SDK (>=1.3)](https://vulkan.lunarg.com/sdk/home)
- [CMake >= 3.20](https://cmake.org/download/)
- [Git to fetch content](https://git-scm.com/)

For the GUI I am using:

-[Clay for the UI layout](https://www.nicbarker.com/clay)
-[Native File Dialog Extened for simple file selection](https://github.com/btzy/nativefiledialog-extended)
-[STB to render fonts and parse images](https://github.com/nothings/stb)

For the **Simple** version of the application everything is done with custom vulkan code using **Compute Shaders** written in `GLSL`.

For the **Advanced** version of the application Almost all of the Image generation is done using custom **Compute Shaders** but the final image is processed using:

-[RELAX Denoiser from NRD](https://github.com/NVIDIA-RTX/NRD)
-[NRI to integrate NRD into the application](https://github.com/NVIDIA-RTX/NRI)
-[FFX to upscale the rendered image before presenting](https://gpuopen.com/fidelityfx-superresolution/)

## Important Note

===

The Scope of the project as presented in the School subject was Far excided, a simple *Rasterization* would have suficed yet i Set out to make a Full *PBR* with *Path Tracing*

nevertheless the School Deadlines applied. At the same time I found that the difficulty of the new scope set by me was substential and while the project got to an "Acceptable" state, Due to Deadlines the project had to be left in an *"unfinished"* state.

This *Unfinished* state means that the Advanced version of the app does not correctly render the final image. while still images from a far might look "Good enough" the simple truth is that they are simply *Incorrect*. 
As far as I know the issue is incorrect integration of NRD Relax Denoising. I had a couple of attempts to fix the issue but at the end came too close to the project deadline and as i have far overshot the requirements of the project I simply had to cut my losses short and validate the project. 

Even through I failed to generate correct *PBR Path Traced* images in real time the project still validated with maximum score of 125.

---

ThereFore I suggest to read the [Warnings](#warning) section and [Future Updates](#possible-future-upgrades) section to get a more realistic idea of the state of this app

For those who might want to see my approach to making a *real time Path Tracer* you can read the [Theory of Raytracing](#theory-of-raytracing) for a short accelerated introduction to the theory behind the math and then see [performance and benchmarking](#performance-and-benchmarking) to see what I did to implement and optimise the app so it could run in real time.

## Warning

## How to Build and Run

## Basic Usage of the Application

## Configuration

## Controls

## Theory of Raytracing

## Assets and Shaders

## Notes about the Code and Optimizations

## Testing and Validation

## Performance and Benchmarking

## Possible Future Upgrades

## Troubleshooting

## Contributing

## License
