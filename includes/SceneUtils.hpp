#pragma once

#include "Object.hpp"

#include <cmath>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/common.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>




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

inline AffineMatrix makeAffine(const glm::mat3& matrix, const glm::vec3& vector)
{
    return AffineMatrix{matrix[0], matrix[1], matrix[2], vector};
}

inline AffineMatrix affineIdentity() {
    return {
        glm::vec3(1,0,0),
        glm::vec3(0,1,0),
        glm::vec3(0,0,1),
        glm::vec3(0,0,0)
    };
}

inline glm::mat3 affineLinear(const AffineMatrix& M)
{
    return glm::mat3(M.collumn_0, M.collumn_1, M.collumn_2);
}

inline glm::vec3 affineTranslation(const AffineMatrix& M)
{
    return M.collumn_t;
}

inline AABB affineTransformAABB(const AffineMatrix& M, const AABB& b)
{
    const glm::mat3 A = affineLinear(M);
    const glm::vec3 t = affineTranslation(M);

    const glm::vec3 c = 0.5f * (b.min + b.max);
    const glm::vec3 e = 0.5f * (b.max - b.min);

    const glm::mat3 AbsA = glm::mat3{ glm::abs(A[0]), glm::abs(A[1]), glm::abs(A[2]) };
    const glm::vec3 cW   = A * c + t;
    const glm::vec3 eW   = AbsA * e;

    return { cW - eW, cW + eW };
}


inline glm::mat4 affineToMat4(const AffineMatrix& M) {
    glm::mat4 R(1.0f);
    R[0] = glm::vec4(M.collumn_0, 0.0f);
    R[1] = glm::vec4(M.collumn_1, 0.0f);
    R[2] = glm::vec4(M.collumn_2, 0.0f);
    R[3] = glm::vec4(M.collumn_t, 1.0f);
    return R;
}

inline glm::vec3 affineTransformPoint(const AffineMatrix& M, const glm::vec3& p)
{
    const glm::mat3 A = affineLinear(M);
    return A * p + M.collumn_t;
}

inline glm::vec3 affineTransformDirection(const AffineMatrix& M, const glm::vec3& d)
{
    const glm::mat3 A = affineLinear(M);
    return A * d;
}

inline AffineMatrix affineCompose(const AffineMatrix& B, const AffineMatrix& A)
{
    const glm::mat3 Ab = affineLinear(B);
    const glm::mat3 Aa = affineLinear(A);
    const glm::mat3 Ac = Ab * Aa;
    const glm::vec3 tc = Ab * affineTranslation(A) + affineTranslation(B);
    return makeAffine(Ac, tc);
}

inline AffineMatrix affineInverse(const AffineMatrix& M)
{
    const glm::mat3 A  = affineLinear(M);
    const float detA =
        A[0].x * (A[1].y * A[2].z - A[1].z * A[2].y) -
        A[1].x * (A[0].y * A[2].z - A[0].z * A[2].y) +
        A[2].x * (A[0].y * A[1].z - A[0].z * A[1].y);

    if (std::abs(detA) < EPSILON)
        return affineIdentity();

    const glm::mat3 Ainv = glm::inverse(A);
    const glm::vec3 tinv = -Ainv * affineTranslation(M);
    return makeAffine(Ainv, tinv);
}


inline glm::mat3 affineNormalMatrix(const AffineMatrix& M)
{
    const glm::mat3 Ainv = glm::inverse(affineLinear(M));
    return glm::transpose(Ainv);
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

    inline AABB transformAABB()
    {
        AABB worldBoundingBox;
        AffineMatrix matrix = transform.affineTransform();
        glm::vec3 center = 0.5f * (boundingBox.min + boundingBox.max);
        glm::vec3 extent = 0.5f * (boundingBox.max - boundingBox.min);
      
        glm::mat3 lin(matrix.collumn_0, matrix.collumn_1, matrix.collumn_2);
        glm::vec3 translate = matrix.collumn_t;
       
        glm::mat3 Abslin{ glm::abs(lin[0]), glm::abs(lin[1]), glm::abs(lin[2]) };
        glm::vec3 centerWorld = lin * center + translate;
        glm::vec3 extentWorld = Abslin * extent;
        worldBoundingBox.min = centerWorld - extentWorld;
        worldBoundingBox.max = centerWorld + extentWorld;
        return worldBoundingBox;
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


struct ObjectMeshData {
    std::string name;
    SBVH bottomLevelAccelerationStructure;
    std::vector<ImageRGBA8> textures;
    std::vector<MaterialGPU> perMeshMaterials;
};


struct Scene {
    Camera camera;
    std::vector<SceneObject> objects;
    std::vector<ObjectMeshData> meshes;
};

static inline void printVec3(std::ostream& os, const char* label, const glm::vec3& v) {
    os << "  " << label << ": (" 
       << std::fixed << std::setprecision(3)
       << v.x << ", " << v.y << ", " << v.z << ")\n";
}

static inline void printAABB(std::ostream& os, const AABB& b) {
    os << std::fixed << std::setprecision(3);
    os << "  AABB.min: (" << b.min.x << ", " << b.min.y << ", " << b.min.z << ")\n";
    os << "  AABB.max: (" << b.max.x << ", " << b.max.y << ", " << b.max.z << ")\n";
    glm::vec3 size = b.max - b.min;
    os << "  AABB.size:(" << size.x << ", " << size.y << ", " << size.z << ")\n";
}

static inline void printCamera(std::ostream& os, const Camera& c) {
    os << "[Camera]\n";
    printVec3(os, "position", c.position);
    printVec3(os, "target  ", c.target);
    printVec3(os, "up      ", c.up);
    os << "  vfovDeg   : " << c.vfovDeg   << "\n"
       << "  aspect    : " << c.aspect    << "\n"
       << "  nearPlane : " << c.nearPlane << "\n"
       << "  farPlane  : " << c.farPlane  << "\n";
}

static inline void printTexture(std::ostream& os, const ImageRGBA8& tex, size_t idx)
{
    os << "        [" << idx << "] "
       << "Path: \"" << tex.filePath << "\" "
       << "(" << tex.width << "x" << tex.height << ")\n";
}

static inline void printMaterial(std::ostream& os, const MaterialGPU& mat, size_t idx)
{
    os << "        [" << idx << "] "
       << "baseColor_opacity: (" << mat.baseColor_opacity.x << ", "
                                << mat.baseColor_opacity.y << ", "
                                << mat.baseColor_opacity.z << ", "
                                << mat.baseColor_opacity.w << ")\n";
    os << "           F0_ior_rough: (" << mat.F0_ior_rough.x << ", "
                                     << mat.F0_ior_rough.y << ", "
                                     << mat.F0_ior_rough.z << ", "
                                     << mat.F0_ior_rough.w << ")\n";
    os << "           emission_flags: (" << mat.emission_flags.x << ", "
                                       << mat.emission_flags.y << ", "
                                       << mat.emission_flags.z << ", "
                                       << mat.emission_flags.w << ")\n";
    os << "           textureId: (" << glm::floatBitsToUint(mat.textureId.x) << ", "
                                  << mat.textureId.y << ", "
                                  << mat.textureId.z << ", "
                                  << mat.textureId.w << ")\n";
}


static inline void printMesh(std::ostream& os, const ObjectMeshData& m, size_t idx)
{
    os << "  [" << idx << "] Mesh: \"" << m.name << "\"\n";
    os << "     SBVH nodes: " << m.bottomLevelAccelerationStructure.nodes.size() << "\n";
    printAABB(os, m.bottomLevelAccelerationStructure.outerBoundingBox);

    os << "     textures : " << m.textures.size() << "\n";
    for (size_t i = 0; i < m.textures.size(); ++i)
        printTexture(os, m.textures[i], i);

    os << "     materials: " << m.perMeshMaterials.size() << "\n";
    for (size_t i = 0; i < m.perMeshMaterials.size(); ++i)
        printMaterial(os, m.perMeshMaterials[i], i);
}


static inline void printSceneObject(std::ostream& os, const SceneObject& o, size_t idx) {
    os << "  [" << idx << "] SceneObject\n";
    os << "     meshID: " << o.meshID << "\n";
    printVec3(os, "translate", o.transform.translate);
    printVec3(os, "rotation ", o.transform.rotation);
    printVec3(os, "scale    ", o.transform.scale);
    printVec3(os, "rotSpeed ", o.animation.rotationSpeedPerAxis);
    os << "     speedScalar: " << o.animation.SpeedScalar << "\n";
    printAABB(os, o.boundingBox);
}

inline void DumpScene(const Scene& scene, std::ostream& os = std::cout) {
    os << "==== Scene Dump ====\n";
    printCamera(os, scene.camera);

    os << "\n[Meshes] count=" << scene.meshes.size() << "\n";
    for (size_t i = 0; i < scene.meshes.size(); ++i) {
        printMesh(os, scene.meshes[i], i);
    }

    os << "\n[Objects] count=" << scene.objects.size() << "\n";
    for (size_t i = 0; i < scene.objects.size(); ++i) {
        printSceneObject(os, scene.objects[i], i);
    }

    os << "==== End Scene Dump ====\n";
}
