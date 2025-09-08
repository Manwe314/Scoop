cmake_minimum_required(VERSION 3.20)
project(Scoop LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Vulkan ---
# Make sure you "source $VULKAN_SDK/setup-env.sh" before running CMake so this finds the SDK.
find_package(Vulkan REQUIRED)

include(FetchContent)
# --- GLFW (static, no install, built in-tree) ---
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- GLM (header-only, no sudo) ---
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glm)

# Shader compile using SDK's glslc
if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK environment not set. Source the SDK first.")
endif()

# --- Find glslc robustly ---
# Try PATH first (setup-env.sh usually adds .../x86_64/bin), but also hint with VULKAN_SDK.
find_program(GLSLC
  NAMES glslc
  HINTS
    "$ENV{VULKAN_SDK}/bin"
    "$ENV{VULKAN_SDK}/x86_64/bin"
  DOC "Path to glslc (Vulkan shader compiler)"
)

if(NOT GLSLC)
  message(FATAL_ERROR "Could not find 'glslc'. Install the Vulkan SDK or ensure 'glslc' is on PATH.")
else()
  message(STATUS "Using GLSLC at: ${GLSLC}")
endif()

# Track shaders and reconfigure when files are added
file(GLOB SHADERS
     CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/shaders/*.vert"
     "${CMAKE_SOURCE_DIR}/shaders/*.frag")

set(SPIRV_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_OUT_DIR})

set(SPV_BINARIES "")
foreach(SRC ${SHADERS})
  get_filename_component(fname ${SRC} NAME)
  set(outspv ${SPIRV_OUT_DIR}/${fname}.spv)
  add_custom_command(
    OUTPUT ${outspv}
    COMMAND ${GLSLC} -g -O ${SRC} -o ${outspv}
    DEPENDS ${SRC}
    BYPRODUCTS ${outspv}          # helps multi-config generators
    COMMENT "Compiling ${fname}"
  )
  list(APPEND SPV_BINARIES ${outspv})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPV_BINARIES})
# --- Sources: all .cpp under src/ (recursively) ---
file(GLOB_RECURSE APP_SOURCES
     CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(scoop ${APP_SOURCES})
add_dependencies(scoop Shaders)

# Header search paths
target_include_directories(scoop
  PRIVATE
    "${CMAKE_SOURCE_DIR}/includes"
    "${CMAKE_SOURCE_DIR}/external"
)

# Link as before
target_link_libraries(scoop PRIVATE Vulkan::Vulkan glfw glm::glm)


# On some distros you may need to link X11 stuff explicitly; GLFW usually handles this.
if(UNIX AND NOT APPLE)
  target_link_libraries(scoop PRIVATE dl pthread)
endif()
