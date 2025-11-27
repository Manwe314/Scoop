#pragma once

#include "Utils.hpp"



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
    float pad[2];
};

struct alignas(16) LightSample {
    glm::vec4 dir;
    glm::vec4 Li;
    float pdf;
    float cosTheta;
    uint32_t lightId;
    uint32_t pad[2];
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
