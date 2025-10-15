#pragma once

#include "Object.hpp"

#include <cmath>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <glm/glm.hpp>



inline float deg2rad(float d)
{
    return d * 3.14159265358979323846f / 180.0f;
}

inline glm::mat3 rotX(float deg)
{
    float cos = std::cos(deg2rad(deg));
    float sin = std::sin(deg2rad(deg));

    return glm::mat3{
        { 1,   0,   0 },
        { 0, cos, sin },
        { 0,-sin, cos }
    };
}

inline glm::mat3 rotY(float deg) {
    float cos = std::cos(deg2rad(deg));
    float sin = std::sin(deg2rad(deg));
    return glm::mat3{
        {  cos, 0,-sin },
        {  0,   1,   0 },
        {  sin, 0, cos }
    };
}

inline glm::mat3 rotZ(float deg) {
    float cos = std::cos(deg2rad(deg));
    float sin = std::sin(deg2rad(deg));
    return glm::mat3{
        { cos, sin, 0 },
        {-sin, cos, 0 },
        { 0,    0,  1 }
    };
}

inline glm::mat3 scaleby(const glm::vec3& scale) {
    return glm::mat3{
        { scale.x, 0.0f, 0.0f },
        { 0.0f, scale.y, 0.0f },
        { 0.0f, 0.0f, scale.z }
    };
}


struct AffineMatrix {
    glm::vec3 collumn_0;
    glm::vec3 collumn_1;
    glm::vec3 collumn_2;
    glm::vec3 collumn_t;
};

inline AffineMatrix makeAffine(glm::mat3& matrix, glm::vec3& vector)
{
    return AffineMatrix{matrix[0], matrix[1], matrix[2], vector};
}


struct Transform {
    glm::vec3 translate{0.0f};
    glm::vec3 rotation{0.0f};
    glm::vec3 scale{1.0f};

    glm::mat3 rotate() {
        return rotZ(rotation.z) * rotY(rotation.y) * rotX(rotation.x);
    }

    glm::mat3 linear() {
        return rotate() * scaleby(scale);
    }

    AffineMatrix affineTransform() {
        glm::mat3 lin = linear();
        return makeAffine(lin, translate);
    }
};

struct RotationAnimation {
    glm::vec3 rotationSpeedPerAxis{0.0f};
    float SpeedScalar = 0.0f;
};


struct SceneObject {
    uint32_t meshID;
    Transform transform;
    AABB boundingBox;
    RotationAnimation animation;

    inline void transformAABB() {
        AffineMatrix matrix = transform.affineTransform();
        glm::vec3 center = 0.5f * (boundingBox.min + boundingBox.max);
        glm::vec3 extent = 0.5f * (boundingBox.max - boundingBox.min);
      
        glm::mat3 lin(matrix.collumn_0, matrix.collumn_1, matrix.collumn_2);
        glm::vec3 translate = matrix.collumn_t;
       
        glm::mat3 Abslin{ glm::abs(lin[0]), glm::abs(lin[1]), glm::abs(lin[2]) };
        glm::vec3 centerWorld = lin * center + translate;
        glm::vec3 extentWorld = Abslin * extent;
        boundingBox.min = centerWorld - extentWorld;
        boundingBox.max = centerWorld + extentWorld;
    }
};

struct Camera {
    glm::vec3 position {0.0f, 0.0f,  5.0f};
    glm::vec3 target   {0.0f, 0.0f,  0.0f};
    glm::vec3 up       {0.0f, 1.0f,  0.0f};

    float vfovDeg   = 60.0f;
    float aspect    = 16.0f / 9.0f;
    float nearPlane = 0.1f;
    float farPlane  = 1000.0f;

    void basis(glm::vec3& U, glm::vec3& V, glm::vec3& W) const
    {
        W = glm::normalize(position - target);

        glm::vec3 upN = glm::normalize(up);
        if (std::abs(glm::dot(upN, W)) > 0.999f)
            upN = std::abs(W.y) < 0.999f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);

        U = glm::normalize(glm::cross(upN, W));
        V = glm::cross(W, U);
    }

};


struct Scene {
    Camera camera;
    std::vector<SceneObject> Objects;
    std::vector<SBVH> bottomLevelAccelerationStructures;
    std::vector<std::string> textures;
    std::vector<std::vector<MaterialGPU>> perMeshMaterials;
};