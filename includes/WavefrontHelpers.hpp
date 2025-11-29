#pragma once

#include "Utils.hpp"

enum QueueIndex : uint32_t {
    QUEUE_NEW_PATH  = 0,
    QUEUE_LOGIC     = 1,
    QUEUE_MATERIAL  = 2,
    QUEUE_SHADOW    = 3,
};


enum PathState : uint32_t {
    PATH_STATE_INIT        = 0,
    PATH_STATE_NEED_HIT    = 1,
    PATH_STATE_NEED_SHADE  = 2,
    PATH_STATE_NEED_SHADOW = 3,
    PATH_STATE_NEED_MIS    = 4,
    PATH_STATE_DEAD_WRITTEN= 5,
    PATH_STATE_COMPLETE    = 6,
    PATH_STATE_DEAD        = 7,
};

struct alignas(16) PixelStatsGPU {
    glm::vec4 mean;
    glm::vec4 m2;
    uint32_t  sampleCount;
    uint32_t  flags;
    glm::uvec2 _pad;
};

struct AdaptiveCountersGPU {
    uint32_t highVarCount;
    uint32_t rescheduleCounter;
    uint32_t _pad0;
    uint32_t _pad1;
};


struct PathHeader {
    uint32_t pixelIndex;
    uint32_t rngState;
    uint32_t depth;
    uint32_t flags;
};

struct alignas(16) Ray {
    glm::vec4 origin;
    glm::vec4 direction;
};

struct HitIds {
    uint32_t instanceIndex;
    uint32_t primitiveIndex;
};

struct Hitdata {
    float worldT;
    float localT;
    float baryU;
    float baryV;
};

struct alignas(16) RadianceState {
    glm::vec4 throughput;
    glm::vec4 radiance;
    float prevBsdfPdf;
    float pad[3];
};

struct alignas(16) BsdfSample {
    glm::vec4 dir;
    glm::vec4 f;
    float pdf;
    float cosTheta;
    float pdf_L_B;
    float pad;
};

struct alignas(16) LightSample {
    glm::vec4 dir;
    glm::vec4 Li_fL;
    float pdf;
    float cosTheta;
    float pdf_B_L;
    uint32_t lightId;
    uint32_t pad;
};

struct alignas(16) ShadowRay {
    glm::vec4 origin;
    glm::vec4 direction;
    uint32_t lightId;
    uint32_t primitiveId;
    uint32_t pad[2];
};

struct ShadowResult {
    uint32_t visible;
    uint32_t pad[3];
};

struct PathQueue {
    uint32_t count;
    uint32_t pad[3];
};
