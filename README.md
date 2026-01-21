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
    - [Approach](#approach)
    - [Object Parsing](#object-parsing)
    - [Preparing for Frame Generation](#preparing-for-frame-generation)
        - [working With Data](#working-with-data)
        - [Per Frame CPU Work](#per-frame-cpu-work)
    - [GPU Optimizations](#gpu-optimizations)
    - [Final steps](#final-Steps)
- [Final Notes](#final-notes)
- [Images](#images)
- [Sources](#sources)
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

with the time limitation on the C++ side our best friend is `-O3` compile flag that allowes us to not worrie too much about small code optimizations like writing `++i` instead of `i++` or using `.emplace_back()` instead of `.push_back()`. especially we get away with things that I might find out later on during the development process and not needing to update the entire codebase. 

> It's also important to remember that Optimization does not always mean *"The Best Option"* optimization could be an approach that is *"good enough"* 

### Object parsing

As our target becomes *"good enough"*, parsing an `.obj` file of a model with 1 Milion+ tringles in 5-6 seconds for a school project validation becomes enough.
that is why a simple loop and going line by line isgood enough since it achieves our benchmark. In [Object.cpp](./src/Object.cpp) I Tried to save the data in a scalable way by grouping them based on [wavefront obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file) grouping convection. 
most of the parsing is standart C++ i attempted to use more modern C++ types like `std::optional` for those data types that were optional to be provided in the `.obj`.

Before moving on to building *SBVH* for objects i had to split polygons into trinagles to store the object as purely tringles. for this i used the [Ear Clipping](https://nils-olovsson.se/articles/ear_clipping_triangulation/#citation-chazelle1991) algorithm. this is a perfect example of good enough. **Ear clipping algorthim** has a big **O** of `N²` by no means is this the *"best"* algorthm but for the size of polygons i expect to be used on this program Ear Clipping is just enough.

Unlike triangulation Building an **SBVH** Is something that i would rather have somewhat Optimized.
For the **Simple** version of the app I would like to have at least an option to run 1 Milion Trinagles in real time so building an SBVH for something that big in reasnoble time is a target. luckly the [Nvidia's original Paper](https://www.nvidia.in/docs/IO/77714/sbvh.pdf) provides an approach called *chopped binning* that allowes us to constuct the **SBVH** with at each "cut" having to do only two `O(N)` passes.

Even though Memory is the one resource we have to spare beeing to careless with it will result with running out of it quickly. Therefore optimizing slightly for it, especially when its simple to do so, is a no brainer. this is why i seperated the **"Instance data"** and **"mesh data"**. In [SceneUtils.hpp](./includes/SceneUtils.hpp) we can see the structs: **Scene**, **SceneObject** and **ObjectMeshData**. this way we only keep one copy of all the materials, trinagles and SBVH nodes per **Unique** `.obj` we parse keeping the memory usage in check. we can refuse to delete this memory so future constuction / loading (untill the application is quit) is performed super fast. we use this all the way to GPU Compute Pipeline where we only upload mesh data **Once Per OBJ** and utilize a mapping variable that allowes multiple instances to use the same mesh data.

### Preparing for Frame Generation

#### Working With Data

Before we start working on frames we must prepare the data for the full CPU & GPU cycle. the first thing I did was to identify and group data. 
Initial seperation is based on the frequency of use. object mesh data is static for the lifetime of the application so we can upload **all** mesh data to the GPU once, before we even start the lifecyle loop. Then we have data like the **TLAS** that needs to be recomputed and uploaded once **every frame**. at the begining of the frame before we pass on to the GPU we must format this data and upload it to the GPU. Lastly we have some data that needs to be moved around pipelines **multiple** times per frame, this is due to the GPU proggraming approach called [WaveFront Path Tracing](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf).

When prepering data, especially the *"static"* data, I tried to have it optimally formated. First keeping structs 16 alignable is core for GPU layout reasons for std430 layout that I am using. Secondly I have to try to group it so, that whenever i fetch data all the locally relevant data comes within the the same fetch. this is due to how [cache lines](https://en.algorithmica.org/hpc/cpu-cache/cache-lines/) work. Basically whenever we request some data from memory we never get exactly the amount of data we requested but rather we get a constant amount of data like 64 or 128 bytes. if the struct we want is 50 bytes we still get 128 bytes first 50 of which will be the struct we requested and next 78 bytes will have whatver is next in memory (likely next entries of structs in our buffer). Because of this its best to keep structs small and best have the size multiple of 2 and 16. Just sizing is not enough though. we have to write data in the buffer such that the additionally fetched data will become relevant on the next loop cycle, otherwise the additionally fetched data will do us no good. 
There 

In the GPU world we have one more additional consideration with how we bundle our data, [register preassure](https://modal.com/gpu-glossary/perf/register-pressure). beyond the chaches each thread we spin up has registers we use registers to hold primitive variables for fast access. but these memory cells on the phicial device are limited so if we use A lot of variables **Throughout** the lifetime of the compute shader we will force the device to use more chache and memory transfers because it cant keep all the variables in registers.

> note that the issue is not just total amount. It's the amount we need at a given moment. we can have 25 vec3's but if we use 5 at a time and in such a way that once used they are not needed for further execution the registers will be freed to load the next 5 vecotrs. But instead if we use these vectors in such a way that they are always used throught the shader the GPU will try to keep them in registers because they will be needed in the future. 

Because of this I attempted to keep most of my stucts 64 bytes and keep the data in them localy relevant so a fetch of a single entry gives me the data that will all be relevant during one logical step of the compute shader. Data that will be needed in a cycle like trinagle vertex data is contiguasly layed in memory so all the relevant trinagle data is fetched optimally.

#### Per Frame CPU Work

Some data has to be Created on the fly before each frame. for example **TLAS** has to be recomputed each frame because objects rotate and/or move in the scene.
My initial though was to use same optimized tactics as I did for the **SBVH** but I realised that due to the GUI and limit with Trinagle counts i have already limited the overall instances to 16. so here using a method thats `O(N²)` at worst will be 256 times worse. thats seems like a lot but the actual work of making a **TLAS** is blazing fast in the order of micro seconds. having a budget of 16.6 miliseconds we can spare 256 **micro** seconds in the **WORST** case.
So in order to optimize for development time is opted to use recursive method to construct the **TLAS**. 

One advantage of having to work with raytracing in real time is that we can have movement and rotations by simply updating the [transformation matrix](https://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/) of each instance. Instead of working with mesh data, we keep all the mesh data in local, model space. before we build the **TLAS** we can simply recalculate the transformation matrix for each instance with the new rotation and position.

The vast majority of the Time spent doing work for each frame is work done on the GPU. because of this we have the CPU mostly idle. we can utilize it better if we allowed it to start work on the next frame before this frame is fully presented. because of this we have in [swapchain.hpp](./includes/SwapChain.hpp) `static constexpr int MAX_FRAMES_IN_FLIGHT = 2;` so we can have some work done on 2 frames at once. You might think it would be better to have this number be higher but that actually wont fly (yes this was intentional :P). even if we allowed to have work done on more frames each frames work still **has** to be sequential so we must orchestrate it with semaphores and fences. also for real time applications we can get unwanted frames because some input might come in on a frame that is displayed and the user will expect changes in the next couple of frames but if we already started work on the next few frames this inputs change will be delayed. 

### GPU Optimizations

I already talked about data and data transfer which in shader code shows up as simply having logical structure such that variables are used and discarded.

> also having explicit `const in` `inout` and other decorators helps a lot

But by far the biggest optimiziation is to implement [wavefront Pathtracing](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf). with wavefront instead of having one compute shader kernel that does all the logic per thread we have multiple kernels that are dispatched in sequance by the CPU. One might think that the overhead of CPU to GPU switching will not make this worth but due to register preassure and something called **SIMT** this is not the case.

**SIMT** or Single Instruction Multiple Threads, is how GPU's ustilize their paralelising speed. basically in a workgroup multiple threads try to execute the same single instruction every clock cycle. This means that if we have multiple threads ina  work group at diferent stages of work many of these threads will have to wait idle while a given instruction is beeing executed. Ray Tracing inherantly is a banching operation so having one mega kernel causes workgroup threads to diverge quite fast and because of this many of the threads are forced to wait for their instructions causing us to loose some of the paralellism of the GPU.
contrast this to multiple kernels we can split the raytracing proccess into logical groups such that at each kernel we can have minimal divergance. utilizing more of the **SIMT** architecture. we also lower register preassure since each kernel now has to operate on a specific chunk of data so less variables might be needed down the line. 

For this project after switching to Wavefront i saw 81% decrease in frame generation time. While this is already good we still need further optimizations to get to real time.

> and remember all these data / architecture optimizations are **In Addition** to the Optimizations outlined in the [theory of Raytracing](#theory-of-raytracing) and even still we are not close to real time path tracer

Using a GPU Profiler I saw that most of GPU work is time spent on tracing rays through the scene and looking up emmisive trinagles. solving the latter was much simpler to get an `O(N)` processing and `O(1)` match time to have trinagles primitive index mapped to its emmisive index i used the [Alias Method](https://en.wikipedia.org/wiki/Alias_method). Luckly Triangle ray and axis aligned bounding box, ray intersection methods are heavly reaserched and optimised so it was as simple as implementing the [Möller-Trumbore](https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf) algorthim for ray trinagle intersection tests and [Slab Method](https://en.wikipedia.org/wiki/Slab_method) for AABB ray intersection tests. 

> note that Major part of ray tracing speedup is the **BLAS** and **TLAS** having these structures optimized gives the greatest performance boost that is why i went with [SBVH](https://www.nvidia.in/docs/IO/77714/sbvh.pdf) as outlined in previous sections.

### Final Steps

After all that optimization what we find is that we are still very very very far away from having a good real time path tracer. the biggest issue is the amount of samples we have to take and the amouont of pixels we have to work with.

Consider this, for a standard 1920 by 1080 image we have 2073600 pixels. even with our best theoretical models we need 30+ samples for something passable when squinting your eyes, thats 62 208 000 rays. but each ray also bounces on avarage 5-6 times. Thus we get 373 248 000 seperate instances where we must determine the fate of a ray. Each time we might need a combined check of AABB and trinagle intersections 8-9 times on avarage. 3 359 232 000  is the amount of intersection work we have to **Each frame**. that is just not feasable because here we are not even count all the work thats not ray intersections and that 30 sample per pixel is a very lowball number.

this is why I had to do 2 extra things. first i had to use **[NRD Relax](https://github.com/NVIDIA-RTX/NRD)** Nvidia's real time denoiser and an upscaler like **[ffx](https://gpuopen.com/fidelityfx-superresolution/)**. to start with the latter, in our previous calculation the first multiplication was the largest we can win big initially if we just render smaller images. halving the width and heigh cuts the pixel count by 75% simply less pixels = less work. But! we still want a full image so we can then Upscale the smaller image to fit a larger canvas. simply *"making pixels bigger"* gives us a lower quality image we have to upscale in a smart way. this is what ffx from AMD does. Next big number we used was the sample count. if we could keep that low suddenly we can make real time feasable. Nvidia's Relax denoiser uses multi dimensional data to achive amazing results with low SPP counts such as 1. so we could sample each pixel once and use NRD to denoise get a good clear image at quorter scale and then use ffx to upscale to full image size.

This is in fact what I am using. I wont go into detail on how NRD works especially because this is also the step I failed at and could not resolve in time, butafter all this work and optimizations we have arrived at real time Path tracer!


## final Notes

In its escence while the scool project is ment to be a quick introduction to 3D and rasterization I used the project to get aquanted with a graphics API like vulkan, C++ aplication coding with using Clay and implementing a GUI and most importantly learn a lot about a specific high level rendering teqnique like Path tracing. Even though the final version of the application is not a fully working program in terms of these goals I'd say that this project was a total success in these 3 months I learned more than in any 3 month streach. That beeing said this does not mean that i do not have a list of Upgrades and changes I'd implement and nor does it mean that i wont ever come back to it, but for now that is not the plan.

heres a taste of that list:

- upgrade Font handling with vulkan and clay
- upgrade vulkan renderer for clay
- add support for more instances
- compleate SBVH building and further optimize buffer creation
- add more anti aliasing support
- support more `.obj` and `.mtl` directives
- add nonuniform scaling support
- add skybox support
- **FIX THE DENOISING ISSUE**

and many more...


## Images

[conell box(s)](./assets/pictures/denoiseconrell.png)
[conell box(s)](./assets/pictures/cornell42.png)
[conell box(s)](./assets/pictures/conellblue.png)
[High SPP](./assets/pictures/HighSPP.png)
[denoising input](./assets/pictures/DenoiseInput.png)
[GUI](./assets/pictures/GUI.png)
[simple versions](./assets/pictures/SimpleRender.png)
[simple versions](./assets/pictures//SimpleMultipleObjects.png)

## sources

1. Möller, T., & Trumbore, B. (1997). Fast, Minimum Storage Ray–Triangle Intersection.
   https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf

2. Stich, M., Friedrich, H., & Dietrich, A. (2013). Spatial Splits in Bounding Volume Hierarchies. NVIDIA.
   https://www.nvidia.in/docs/IO/77714/sbvh.pdf

3. Olovsson, N. (n.d.). Ear Clipping Triangulation.
   https://nils-olovsson.se/articles/ear_clipping_triangulation/#citation-chazelle1991

4. Shirley, P., et al. (2018). Ray Tracing in One Weekend.
   https://raytracing.github.io/books/RayTracingInOneWeekend.html

5. Mitchell, D. P., & Netravali, A. N. (1988). Reconstruction Filters in Computer Graphics.
   https://www.graphics.cornell.edu/pubs/1997/MT97.pdf

6. OpenGL Tutorial – Matrices.
   https://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/

7. Wavefront OBJ File Format. Wikipedia.
   https://en.wikipedia.org/wiki/Wavefront_.obj_file

8. Affine Transformation. Wikipedia.
   https://en.wikipedia.org/wiki/Affine_transformation

9. Mipmap. Wikipedia.
   https://en.wikipedia.org/wiki/Mipmap

10. Pharr, M., Humphreys, G., & Jakob, W. (2016). Physically Based Rendering: From Theory to Implementation.
    https://graphics.stanford.edu/papers/trd/trd_jpg.pdf

11. https://www.pbr-book.org/4ed/Shapes/Triangle_Meshes

12. Ray Tracing Lecture Series. YouTube Playlist.
    https://www.youtube.com/playlist?list=PLmIqTlJ6KsE2yXzeq02hqCDpOdtj6n6A9

13. Stanford CS348B (2018). Reflection and Refraction.
    https://graphics.stanford.edu/courses/cs348b-18-spring-content/lectures/12_reflection2/12_reflection2_slides.pdf

14. Veach, E. (1997). Robust Monte Carlo Methods for Light Transport Simulation. Chapter 9.
    https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf

15. Uwa4d. (2018). Physically Based Rendering: GGX Microfacet Distribution.
    https://medium.com/@uwa4d/physically-based-rendering-more-accurate-microsurface-distribution-function-ggx-3968fc09fa48

16. Physically Based Rendering (PBR) Overview.
    https://graphicscompendium.com/gamedev/15-pbr

17. Li, et al. (2023). Advances in Path Tracing.
    https://arxiv.org/pdf/2306.05044

18. Random Number Generation in GLSL.
    https://amindforeverprogramming.blogspot.com/2013/07/random-floats-in-glsl-330.html

19. Shi, Y. (n.d.). Mipmapping and Anisotropic Filtering.
    https://shi-yan.github.io/webgpuunleashed/Basics/mipmapping_and_anisotropic_filtering.html

20. Walker, A. J. (1977). An Efficient Method for Generating Discrete Random Variables with General Distributions.
    https://en.wikipedia.org/wiki/Alias_method

21. Veach, E., & Guibas, L. J. (1995). Optimally Combining Sampling Techniques for Monte Carlo Rendering.
    https://dl.acm.org/doi/pdf/10.1145/218380.218500

22. https://dl.acm.org/doi/pdf/10.1145/218380.218498

23. Novák, J., et al. (2014). Multiple Importance Sampling.
    https://cw.fel.cvut.cz/b241/_media/courses/b4m39rso/tutorials/multiple_importance_sampling_101.pdf

24. Laine, S., et al. (2013). Megakernels Considered Harmful. NVIDIA Research.
    https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf

25. Chen, C. (n.d.). Welford’s Online Algorithm.
    https://changyaochen.github.io/welford/

26. Schütte, J. (n.d.). Area Light Sampling.
    https://schuttejoe.github.io/post/arealightsampling/

27. Stanford CS348B (2022). Direct Lighting.
    https://gfxcourses.stanford.edu/cs348b/spring22content/media/directillum/directlighting1.pdf

28. Xu, Z. (2013). Monte Carlo Integration.
    https://www3.nd.edu/~zxu2/acms40390F13/MC-Integration.pdf

## License

- License: The contents of this repository are licensed as [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- Authors: I am the sole author [see my git](https://github.com/Manwe314)