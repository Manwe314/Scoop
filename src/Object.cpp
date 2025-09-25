#include "Object.hpp"

enum {
    VERTEX,
    TEXTURE,
    NORMAL,
    MTL_FILE,
    OBJ_NAME,
    GROUP_NAME,
    MATERIAL,
    FACE,
    SMOOTH_SHADING,
};

std::optional<float> Object::to_float(const std::string& s)
{
    float value{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec == std::errc())
        return value;
    return std::nullopt;
}


Object::Object(std::string &filePath) : vertices(), textureCoords(), normals()
{
    std::ifstream file(filePath);
    if (!file.is_open())
        throw std::runtime_error("Could not open the .obj file");

    std::string line;
    while (std::getline(file, line))
    {
        int type = -1;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (iss >> token)
            tokens.push_back(token);

        if (!tokens.empty() && tokens[0][0] == '#')
            continue;
        if (tokens[0] == "v")
            type = VERTEX;
        else if (tokens[0] == "vt")
            type = TEXTURE;
        else if (tokens[0] == "vn")
            type = NORMAL;
        else if (tokens[0] == "mtllib")
            type = MTL_FILE;
        else if (tokens[0] == "o")
            type = OBJ_NAME;
        else if (tokens[0] == "g")
            type = GROUP_NAME;
        else if (tokens[0] == "usemtl")
            type = MATERIAL;
        else if (tokens[0] == "f")
            type = FACE;
        else if (tokens[0] == "s")
            type = SMOOTH_SHADING;
        
        parseLine(tokens, type);
    }

}

Object::~Object()
{

}


void Object::parseLine(std::vector<std::string> tokens, int type)
{
    switch (type)
    {
    case VERTEX:
        if (tokens.size() < 4)
            throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Vertex");
        glm::vec3 vertex{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Vertex Float");
            vertex[i] = *f;
        }
        vertices.push_back(vertex);
        break;
    case TEXTURE:
        if (tokens.size() < 2)
            throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Texture Coordinate");
        glm::vec3 textureCoord{};
        for (int i = 0; i < 3; ++i)
        {
            if (i + 1 >= tokens.size())
                textureCoord[i] = 0.0;
            else
            {
                auto f  = to_float(tokens[i + 1]);
                if (!f)
                    throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Texture Coordinate Float");
                textureCoord[i] = *f;
            }
        }
        textureCoords.push_back(textureCoord);
        break;
    case NORMAL:
        if (tokens.size() < 4)
            throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Normal");
        glm::vec3 normal{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Normal Float");
            normal[i] = *f;
        }
        normals.push_back(normal);
        break;
    case MTL_FILE:
        
        break;
    case OBJ_NAME:
        
        break;
    case GROUP_NAME:
        
        break;
    case MATERIAL:
        
        break;
    case FACE:
        
        break;
    case SMOOTH_SHADING:
        
        break;
    default:
        throw std::runtime_error("Unknown line definition found in the OBJ file");
        break;
    }

}