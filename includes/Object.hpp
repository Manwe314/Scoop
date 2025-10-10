#pragma once

#include "Utils.hpp"

#include <iostream>
#include <vector>
#include <array>
#include <glm/glm.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <charconv> 
#include <system_error>
#include <optional>
#include <algorithm>
#include <numeric>
#include <map>
#include <unordered_map>

struct Bin
{
    AABB     bb;
    uint32_t count;
};

struct BuildTask {
    uint32_t first;
    uint32_t count;
    int      parent;
    int      depth;
    bool     asLeft;
    float    rootSA;
};

struct SplitResult {
    bool      valid;
    bool      spatial;
    int       axis;
    float     pos;
    float     cost;
    AABB      leftBB;
    AABB      rightBB;
    uint32_t  leftCount;
    uint32_t  rightCount;
};

struct alignas(16) ShadingTriangle {
    glm::vec4 vertNormal0_uv;
    glm::vec4 vertNormal1_uv;
    glm::vec4 vertNormal2_uv;
};
static_assert(sizeof(ShadingTriangle) == 48);


struct alignas(16) MollerTriangle {
    glm::vec4 vertex_0;
    glm::vec4 edge_vec1;
    glm::vec4 edge_vec2;
    glm::vec4 normal_mat;
};
static_assert(sizeof(MollerTriangle) == 64, "Moller Triangle must be 64 bytes");

struct alignas(16) BVHNode {
    glm::vec3 bbMin_left;
    glm::vec3 bbMax_left;
    glm::vec3 bbMin_right;
    glm::vec3 bbMax_right;
    uint32_t left_start;
    uint32_t left_count;
    uint32_t right_start;
    uint32_t right_count;
};
static_assert(sizeof(BVHNode) == 64, "BVH Node must be 64 bytes");

struct SBVHNode
{
    glm::vec4 bbMin_left__left_start;
    glm::vec4 bbMax_left__left_count;
    glm::vec4 bbMin_right__right_start;
    glm::vec4 bbMax_right__right_count;
};
static_assert(sizeof(SBVHNode) == 64, "BVH Node must be 64 bytes");

struct Triangles
{
    std::vector<MollerTriangle> intersectionTriangles;
    std::vector<ShadingTriangle> shadingTriangles;
};


struct SBVH {
    std::vector<SBVHNode> nodes;
    Triangles triangles;
    AABB outerBoundingBox;
};

struct Face {
    std::array<int, 3> vertices;
    std::optional<std::array<int, 3>> textureCoords;
    std::optional<std::array<int, 3>> normals;
    uint32_t smoothingGroup;
    std::optional<std::string> material;
};

struct Group {
    std::string name;
    std::vector<Face> faces;
};

struct SubObject {
    std::string name;
    std::vector<Group> groups;
};

struct TriRef {
    Face* referance;
    AABB boundingBox;
};

struct BuildState {
    std::vector<TriRef>  refs;
    std::vector<BVHNode> nodes;
};

struct alignas(16) MaterialGPU {
    glm::vec4 baseColor_opacity;
    glm::vec4 F0_ior_rough;
    glm::vec4 emission_flags;
    glm::vec4 futuretextures;
};
static_assert(sizeof(MaterialGPU) == 64, "MaterialGPU must be 64 bytes");

struct Material {
    std::string name;

    glm::vec3 albedo;
    glm::vec3 reflectance;
    glm::vec3 emission;
    
    float shininess;
    float opacity;
    float refractive_index;

    std::optional<std::string> texture;
    std::optional<glm::vec3> texture_offset;
    std::optional<glm::vec3> texture_scale;
};

struct VertexNormalData
{
    int id;
    std::map<Face*, glm::vec3> adjacentNormal;
};

struct PartitionResult {
    uint32_t mid;
    uint32_t leftCount;
    uint32_t rightCount;
    uint32_t rightStart;
    AABB     leftBB;
    AABB     rightBB;
};


class Object
{
private:
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> textureCoords;
    std::vector<glm::vec3> normals;
    std::vector<std::string> mtlFilePaths;
    std::string objFilePath;

    std::vector<SubObject> objects;
    std::map<std::string, Material> materials;
    std::unordered_map<std::string, uint32_t> matNameToId;
    uint32_t nextMatId = 0;

    void parseLine(std::vector<std::string> tokens, int type);
    void parseMaterialLine(std::vector<std::string> tokens, int type, std::string filePath);
    void parseFace(std::vector<std::string> tokens);
    void loadMaterials();
    void addNormals();
    Triangles constructTriangles(BuildState& state);
    MollerTriangle makeMollerTriangle(TriRef& ref);
    ShadingTriangle makeShadingTriangle(TriRef& ref);
    SplitResult findBestObjectSplit(const std::vector<TriRef>& referances, uint32_t first, uint32_t count,const AABB& centroidBB, const AABB& nodeBB);
    SplitResult findBestSpatialSplit(const std::vector<TriRef>& referances, uint32_t first, uint32_t count, const AABB& centroidBB ,const AABB& nodeBB); 
    std::optional<float> to_float(const std::string& s);
    std::vector<TriRef> getRefArray() const;
    uint32_t getMaterialId(const std::optional<std::string>& optName);
    
    std::string currantGroup = "";
    std::string currantObject = "";
    std::string currantMeterial = "";
    uint32_t currantSmoothing = 0;
    
    std::string currantMTLmat = "";
    size_t total_faces = 0;
    
    int reading_line = 1;
    
    inline int findObject(std::string& name)
    {
        if (objects.empty())
        return -1;
        for (int i = 0; i < objects.size(); i++)
        if (objects[i].name == name)
        return i;
        return -1;
    }
    
    inline int findGroup(std::string& name, SubObject& obj)
    {
        if (obj.groups.empty())
        return -1;
        for (int i = 0; i < obj.groups.size(); i++)
        if (obj.groups[i].name == name)
        return i;
        return -1;
    }
    
    inline VertexNormalData findVertexData(int id, std::vector<VertexNormalData>& vertexData)
    {
        for (auto& data : vertexData)
            if (data.id == id)
                return data;
        VertexNormalData vertex{};
        vertex.id = id;
        return vertex;
    }
    
    
    public:
    Object();
    Object(const std::string &filePath);
    ~Object();
    SBVH buildSplitBoundingVolumeHierarchy();
    std::vector<MaterialGPU> buildMaterialGPU();
    Material getDefaultMaterial();
};


