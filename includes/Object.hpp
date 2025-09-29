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
    std::vector<Group> groups;
    std::string name;
};

struct Material {
    std::string name;
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

    void parseLine(std::vector<std::string> tokens, int type);
    std::optional<float> to_float(const std::string& s);

    std::string currantGroup = "";
    std::string currantObject = "";
    std::string currantMeterial = "";
    uint32_t currantSmoothing = 0;


public:
    Object(std::string &filePath);
    ~Object();
};


