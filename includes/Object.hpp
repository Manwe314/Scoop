#pragma once

#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <charconv> 
#include <system_error>
#include <optional>

class Object
{
private:
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> textureCoords;
    std::vector<glm::vec3> normals;

    void parseLine(std::vector<std::string> tokens, int type);
    std::optional<float> to_float(const std::string& s);


public:
    Object(std::string &filePath);
    ~Object();
};


