#pragma once

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

    void parseLine(std::vector<std::string> tokens, int type);
    void parseMaterialLine(std::vector<std::string> tokens, int type, std::string filePath);
    void parseFace(std::vector<std::string> tokens);
    void loadMaterials();
    void addNormals();
    std::optional<float> to_float(const std::string& s);

    std::string currantGroup = "";
    std::string currantObject = "";
    std::string currantMeterial = "";
    uint32_t currantSmoothing = 0;

    std::string currantMTLmat = "";

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
    Object(std::string &filePath);
    ~Object();
    Material getDefaultMaterial();
};


