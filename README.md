# Scoop

This is a school 42 project about parsing and rendering a 3D obj file in 3D space. 
As the scope of the project dictates there is a free choice of programing language and graphics API
I went with **C++** and **Vulkan** writing shaders in **GLSL**
The Project also does not limit us in the rendering methodology although clasic raseterization is recomended
Netherless I went with RayCasting and Full PathTracing as my rendering strategy.
I also added a GUI app layer to the program to help set up scenes.

## Table of Contents

- [Project Overview](#project-overview)
    - [Introduction](#introduction)
    - [Dependencies and Technologies](#Dependencies-and-Technologies)
- [Important Note](#important-note)
- [Warning](#warning)
- [Using The Application](#using-the-application)
    - [Quick Start](#quick-start)
    - [Makefile Use](#makefile-use)
    - [application use](#aplication-use)
    - [controls](#controls)
- [Theory of Raytracing](#theory-of-raytracing)
- [Notes about the Code and Optimizations](#notes-about-the-code-and-optimizations)
- [Testing and Validation](#testing-and-validation)
- [Performance and Benchmarking](#performance-and-benchmarking)
- [Possible Future Upgrades](#possible-future-upgrades)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview


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
Scoop was developed using C++20. In escence the application should be fully cross platform and runable from windows, mac and linux but the build system in place is only for linux.

The requirements are:

- [Vulkan SDK (>=1.3)](https://vulkan.lunarg.com/sdk/home)
- [CMake >= 3.20](https://cmake.org/download/)
- [Git to fetch content](https://git-scm.com/)

For the GUI I am using:

- [Clay for the UI layout](https://www.nicbarker.com/clay)
- [Native File Dialog Extened for simple file selection](https://github.com/btzy/nativefiledialog-extended)
- [STB to render fonts and parse images](https://github.com/nothings/stb)

For the **Simple** version of the application everything is done with custom vulkan code using **Compute Shaders** written in `GLSL`.

For the **Advanced** version of the application Almost all of the Image generation is done using custom **Compute Shaders** but the final image is processed using:

- [RELAX Denoiser from NRD](https://github.com/NVIDIA-RTX/NRD)
- [NRI to integrate NRD into the application](https://github.com/NVIDIA-RTX/NRI)
- [FFX to upscale the rendered image before presenting](https://gpuopen.com/fidelityfx-superresolution/)

## Important Note


The Scope of the project as presented in the School subject was Far excided, a simple *Rasterization* would have suficed yet i Set out to make a Full *PBR* with *Path Tracing*

nevertheless the School Deadlines applied. At the same time I found that the difficulty of the new scope set by me was substential and while the project got to an "Acceptable" state, Due to Deadlines the project had to be left in an *"unfinished"* state.

This *Unfinished* state means that the Advanced version of the app does not correctly render the final image. while still images from a far might look "Good enough" the simple truth is that they are simply *Incorrect*. 
As far as I know the issue is incorrect integration of NRD Relax Denoising. I had a couple of attempts to fix the issue but at the end came too close to the project deadline and as i have far overshot the requirements of the project I simply had to cut my losses short and validate the project. 

Even through I failed to generate correct *PBR Path Traced* images in real time the project still validated with maximum score of 125.

---

ThereFore I suggest to read the [Warnings](#warning) section and [Future Updates](#possible-future-upgrades) section to get a more realistic idea of the state of this app

For those who might want to see my approach to making a *real time Path Tracer* you can read the [Theory of Raytracing](#theory-of-raytracing) for a short accelerated introduction to the theory behind the math and then see [performance and benchmarking](#performance-and-benchmarking) to see what I did to implement and optimise the app so it could run in real time.

## Warning

STB's external tools that I use document that there might be security vunrablities, therefore it is best to use the application for educational purposes only.

The program in general should not have a critical shutdown but it does use a lot of memory and has not been rigorusly tested. It is best to use with caution and keep the scenes small.

keep in mind:

- In advanced mode scaling **non** uniformly will not result in correct rendering (keep x/y/z of the scale the same number).
- While setting up the scene it is best to keep instances to a minimum (max 12).
- The program keeps the mesh data and SBVH data of objects that might have been deleted. This is intentional for school ecosystem where the same model might be used many times on diferent runs to keep processing times low. After using the app for many models its best to quit and rerun once in a while to keep memory to a minimum. (additional instances do not cost extra memory).
- Mtl file only uses some of the properties, that can be seen in [object.cpp](./src/Object.cpp) and only **`.png`** files are used for textures.
- The native File Dialogue pop up can potentially cause issues best use its buttons normally.

## Using The Application

### Quick Start 

Assuming you have all the [prerequisites](#dependencies-and-technologies) and you have read [Warrnings](#warning) On *linux* you can simply navigate to the root of the repository on any terminal and type `make run`. 
> the first time you type `make run` you will be prompted with:
>> Vulkan SDK not found
>> Warrning: by pressing Enter you'll only set SDK for make's shell and will have to re source on other runs
>> If you want to set SDK for this sesion press Ctrl+C and then just press Ctrl+Shift+V to run the command manually
> to run the program vulkan SDK must be sourced in the terminal env. you can press enter to source it for only that run or abort with `Ctrl+C` and the new command will be already set in your clipboard so you can just do `Ctrl+Shift+V` to paste the new command that will source the sdk and then do `make run` all in one input.

Before compiling and running go to [ShowcaseApp.hpp](/includes/ShowcaseApp.hpp) and at top you will find: `static constexpr bool SimpleRayTrace = true;` keep it `true` if you want to compile the *simple* version or set to `false` to compile the *advanced* version.
> if you already compiled one version and want to switch you will need to run `make flcean` first and then `make run`.

### Makefile Use

Makefile also exposes:
- `make` will compile the program, the executable will be in the *build* folder named *scoop*
- `make clean` / `make fclean` will remove the compiled files and will delete the build folder.
- `make re` will do `make fclean` and then recompile the program
- `make run` compiles and runs the program in one go
- `make rerun` will first do `make fclean` and then compile and run the program
- `make shaders` will delete compiled shaders and then recompile them

### Aplication Use

On the GUI you can set up the scene and choose which device will run the graphical computing. the options that are compatible will be displayed if you click the by default selected device. 

to load in a model you must type in the relative path to the `.obj` file. (relative to the root of repository)
> for example: ./assets/models/42.obj 
you can also use the native file dialogue to select the `.obj` that way. once seleceted press load model to load the model data into memory.

next at the bottom section you will see tabs. first tab will walways be `camera`. As you load in more objects additional tabs will be made.
> note: try to keep total instances low 8-10 max. and note that the UI renderer is fully custom and might not handle edge cases like text not fitting well.

on the *camera* tab you can:
- set **position** by typing X Y Z coordinates
- set **looking at** target this dictates the direction towards which the camera is looking at initially. 
- set **UP** vector to tell the program which way is considered "UP" by defult.
- set **field of view** in degrees max 180.
- set **aspect rattio**
- set **near** and **far** clipping ranges. at what distances would objects no longer appear on screen.

> note that postion and looking at can be changed with keyboard and mouse after running the renderer, as well the aspect rattio via normal window resizing.

on the *models* tab(s) you can:
- set **position**, **rotation** and **scale** by typing X Y Z components of each.
- set **animation Rotation** speed per axis, meaning each model can rotate along more than one axis at differing speeds
- set **speed multiplier** this value will act as a gneral multiplier of 
- add an instance of this model. the new instnce may have new values for previously mentiond stats but will share the the same mesh.
- delete an instance.

after seting up the model you must click `build SBVH & Materials`. this will build the required structures for this mesh in the background. the `launch` button will be slightly transparent untill all objects SBVH and Materials are built.
> Note that you cant run the main app untill every mesh in the scene has its SBVH built and multiple instances share SBVH's so you only have to build it per unique `.obj` and not per instance.

once the **Launch** button bcomes opaque you can run the application by clicking it. this will close the GUI window and run the Scene. after viewing the rendering you can close the scene window and that will bring back the GUI window for further tweaks. if you want to quit the application simply close the GUI window.

> note: in the **Advanced** version of the app there **must** be at least one trinagle with light emmisive material proporties.

### Controls

for both **simple** and **advanced** versions:
- Use `WASD` to move the camera *left or right* and *front or back*
- Use `Space` and `left Shift` to move *up* and *down* respectivly
- Use `RMB` by clicking and holding to chnage the view direction.
- Use `.` and `,` to cycle object selection. the selected object will have a *yellow* glow
- Use `esc` to deselect objects.
- Use `F & G` to move the selected object *left or right*
- Use `H & J` to move the selected object *up or down*
- Use `K & L` to move the selected object *front or back*
- Use `T & Y` to Rotate the selected object along the *X axsis*
- Use `U & I` to Rotate the selected object along the *Y axsis*
- Use `O & P` to Rotate the selected object along the *Z axsis*

for the **simple** version you can use `B` to Switch between showing the texture applied to the model and showing normals based coloring.
> note for the models that do not have any textures this does nothing.

for the **advanced** version you can use `N` to switch to **NRD**'s own debug view OR use `M` to switch to custom view of the 5 images that we pass to **NRD**.


## Theory of Raytracing

## Notes about the Code and Optimizations

When writing a real time *path tracer* optimizations are inevitable step for any developement.
But many of the optimizations that make real differance are *theory* based, they are optimizations in the approach to the subject rather than optimizations in code. that is why we saw many of them in [Theory of Raytracing](#theory-of-raytracing). Here I'd like to focus on more **code and architecture** optimizations instead.

### Approach

First we must define the real state of the problem at hand. this is not a full production application, it is a school project that must be done within a deadline. 
because we have this school context we can understand what we are actually optimizing for. The first and most important realization is that we have a **Target Device**.
the school computers are the primary devices that must run our program. they have more than *32 GB* of ram and have a dedicated GPU *AMD Radeon Rx 6500*.
Our main goal is to have a **real time** ray tracer that must be developed within a timeframe. Therefore We have to optimize for performance since the GPU is not the best and we must optimize for development time, while our main resource that we have to spare is memory.

with the time limitation on the C++ side our best friend is `-O3` compile flag that allowes us to not worrie too much about small code optimizations like writing `++i` instead of `i++` or using `.emplace_back()` instead of `.push_back()`. especially we get away with things that I might fins out later on the development process by not needing to update the entire codebase. 

> It's also important to remember that Optimization does not always mean *"The Best Option"* optimization could be an approach that is *"good enough"* 

### Object parsing

As our target becomes *"good enough"*, parsing an `.obj` file of a model with 1 Milion+ tringles in 5-6 seconds for a school project validation becomes enough.
that is why a simple loop and going line by line isgood enough since it achieves our benchmark. In [Object.cpp](./src/Object.cpp) I Tried to save the data in a scalable way by grouping them based on [wavefront obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file) grouping convection. 
most of the parsing is standart C++ i attempted to use more modern C++ types like `std::optional` for those data types that were optional to be provided in the `.obj`.

Before moving on to building *SBVH* for objects i had to split polygons into trinagles to store the object as purely tringles. for this i used the [Ear Clipping](https://nils-olovsson.se/articles/ear_clipping_triangulation/#citation-chazelle1991) algorithm. this is a perfect example of good enough. **Ear clipping algorthim** has a big **O** of `N²` by no means is this the *"best"* algorthm but for the size of polygons i expect to be used on this program Ear Clipping is just enough.

Unlike triangulation Building an **SBVH** Is something that i would rather have somewhat Optimized.
For the **Simple** version of the app I would like to have at least an option to run 1 Milion Trinagles in real time so building an SBVH for something that big in reasnoble time is a target. luckly the [Nvidia's original Paper](https://www.nvidia.in/docs/IO/77714/sbvh.pdf) provides an approach called *chopped binning* that allowes us to constuct the **SBVH** with at each "cut" having to do only two `O(N)` passes.

Even though Memory is the one resource we have to spare beeing to careless with it will result with running out of it quickly. Therefore optimizing slightly for it, especially when its simple to do so, is a no brainer. this is why i seperated the **"Instance data"** and **"mesh data"**. In [SceneUtils.hpp](./includes/SceneUtils.hpp) we can see the structs: **Scene**, **SceneObject** and **ObjectMeshData**. this way we only keep one copy of all the materials, trinagles and SBVH nodes per **Unique** `.obj` we parse keeping the memory usage in check. we can refuse to delete this memory so future constuction / loading (untill the application is quit) is performes super fast. we use this all the way to GPU Compute Pipeline where we only upload mesh data **Once Per OBJ** and utilize a mapping variab that allowes multiple instances to use the same mesh data.

### Preparing for Frame Generation

### GPU Optimizations

## Testing and Validation

## Performance and Benchmarking

## Possible Future Upgrades

## Troubleshooting

## Contributing

## License
