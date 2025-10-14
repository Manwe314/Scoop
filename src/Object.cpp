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

enum {
    ALBEDO,
    REFLECTANCE,
    EMISSION,
    SHININESS,
    OPACITY,
    REFRACTIVEID,
    TEXTUREFILE,
    NAME,
};

std::optional<float> Object::to_float(const std::string& s)
{
    float value{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec == std::errc())
        return value;
    return std::nullopt;
}

static std::string trimAfterLastSlashOrBackslash(const std::string& input)
{
    size_t posSlash = input.find_last_of('/');
    size_t posBackslash = input.find_last_of('\\');

    size_t pos = std::max(posSlash, posBackslash);

    if (pos == std::string::npos)
        return "";

    return input.substr(0, pos + 1);
}

static std::vector<std::string> split(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string current;

    for (char c : str)
    {
        if (c == delimiter)
        {
            tokens.push_back(current);
            current.clear();
        }
        else
            current += c;
    }
    tokens.push_back(current);
    return tokens;
}

static inline bool correctFace(Face& face, const std::vector<glm::vec3>& vertices, const std::vector<glm::vec3>& textureCoords, const std::vector<glm::vec3>& normals)
{
    if (face.vertices[0] == 0 || face.vertices[1] == 0 || face.vertices[2] == 0)
        return false;
    if (std::abs(face.vertices[0]) > vertices.size() || std::abs(face.vertices[1]) > vertices.size() || std::abs(face.vertices[2]) > vertices.size())
        return false;
    if (face.textureCoords.has_value())
        if (std::abs((*face.textureCoords)[0]) > textureCoords.size() || std::abs((*face.textureCoords)[1]) > textureCoords.size() || std::abs((*face.textureCoords)[2]) > textureCoords.size())
            return false;
    if (face.normals.has_value())
        if (std::abs((*face.normals)[0]) > normals.size() || std::abs((*face.normals)[1]) > normals.size() || std::abs((*face.normals)[2]) > normals.size())
            return false;
    for (int i = 0; i < 3; i++)
    {
        if (face.vertices[i] < 0)
            face.vertices[i] = vertices.size() - std::abs(face.vertices[i]) + 1;
        if (face.textureCoords.has_value())
            if ((*face.textureCoords)[i] < 0)
                (*face.textureCoords)[i] = textureCoords.size() - std::abs((*face.textureCoords)[i]) + 1;
        if (face.normals.has_value())
            if ((*face.normals)[i] < 0)
                (*face.normals)[i] = normals.size() - std::abs((*face.normals)[i]) + 1;
    }
    return true;
}

Object::Object()
{

}

Object::Object(const std::string &filePath) : vertices(), textureCoords(), normals(), mtlFilePaths(), objects(), materials(), matNameToId()
{
    
    std::ifstream file(filePath);
    if (!file.is_open())
        throw std::runtime_error("Could not open the .obj file");

    objFilePath = filePath;
    std::string line;
    while (std::getline(file, line))
    {
        int type = -1;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (iss >> token)
            tokens.push_back(token);

        if (tokens.empty())
        {
            reading_line++;
            continue;
        }
        if (!tokens[0].empty() && tokens[0][0] == '#')
        {
            reading_line++;
            continue;
        }
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
        reading_line++;
    }
    loadMaterials();
    addNormals();
}

Object::~Object()
{

}


void Object::parseLine(std::vector<std::string> tokens, int type)
{
    std::string mtl_file;
    std::string temp;
    switch (type)
    {
    case VERTEX: {
        if (tokens.size() < 4)
            throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Vertex. at Line - " + std::to_string(reading_line));
        glm::vec3 vertex{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Vertex Float. at Line - " + std::to_string(reading_line));
            vertex[i] = *f;
        }
        vertices.push_back(vertex);
    }
        break;
    case TEXTURE: {
        if (tokens.size() < 2)
            throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Texture Coordinate. at Line - " + std::to_string(reading_line));
        glm::vec3 textureCoord{};
        for (int i = 0; i < 3; ++i)
        {
            if (i + 1 >= tokens.size())
                textureCoord[i] = 0.0;
            else
            {
                auto f  = to_float(tokens[i + 1]);
                if (!f)
                    throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Texture Coordinate Float. at Line - " + std::to_string(reading_line));
                textureCoord[i] = *f;
            }
        }
        textureCoords.push_back(textureCoord);
        break;
    }
    case NORMAL: {
        if (tokens.size() < 4)
            throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Normal. at Line - " + std::to_string(reading_line));
        glm::vec3 normal{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Normal Float. at Line - " + std::to_string(reading_line));
            normal[i] = *f;
        }
        normals.push_back(normal);
        break;
    }
    case MTL_FILE:
        if (tokens.size() < 2)
            throw std::runtime_error("OBJ file corrupted: MTL file declaration missing the file. at Line - " + std::to_string(reading_line));
        mtl_file = trimAfterLastSlashOrBackslash(objFilePath);
        mtl_file.append(tokens[1]);
        mtlFilePaths.push_back(mtl_file);
        break;
    case OBJ_NAME:
        if (tokens.size() < 2)
            throw std::runtime_error("OBJ file corrupted: Object declaration missing name. at Line - " + std::to_string(reading_line));   
        currantObject = tokens[1];
        break;
    case GROUP_NAME:
        if (tokens.size() < 2)
                throw std::runtime_error("OBJ file corrupted: Group declaration missing name. at Line - " + std::to_string(reading_line));   
            currantGroup = tokens[1];
        break;
    case MATERIAL:
        if (tokens.size() < 2)
            throw std::runtime_error("OBJ file corrupted: Material declaration missing name. at Line - " + std::to_string(reading_line));   
        currantMeterial = tokens[1];
        break;
    case FACE:
        if (tokens.size() < 4)
            throw std::runtime_error("OBJ file corrupted: Face declaration missing vertex(s). at Line - " + std::to_string(reading_line));
        if (!tokens.empty())
            tokens.erase(tokens.begin());
        parseFace(tokens);
        break;
    case SMOOTH_SHADING:
        if (tokens.size() < 2)
            throw std::runtime_error("OBJ file corrupted: Object declaration missing name. at Line - " + std::to_string(reading_line));   
        std::transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), [](unsigned char c){return std::tolower(c); });
        if (tokens[1] == "off")
            currantSmoothing = 0;
        else
        {
            auto f = to_float(tokens[1]);
            if (!f)
                throw std::runtime_error("OBJ file corrupted: an invalid declaration of a Smoothing ID. at Line - " + std::to_string(reading_line));
            currantSmoothing = static_cast<uint32_t>(*f);
        }
        break;
    default:
        throw std::runtime_error("Unknown line definition found in the OBJ file. at Line - " + std::to_string(reading_line));
        break;
    }
}

void Object::parseFace(std::vector<std::string> tokens)
{
    int objIdx = Object::findObject(currantObject);
    if (objIdx < 0)
    {
        objects.push_back(SubObject{currantObject, {}});
        objIdx = static_cast<int>(objects.size()) - 1;
    }
    SubObject& objectRef = objects[objIdx];

    int grpIdx = Object::findGroup(currantGroup, objectRef);
    if (grpIdx < 0)
    {
        objectRef.groups.push_back(Group{currantGroup, {}});
        grpIdx = static_cast<int>(objectRef.groups.size()) - 1;
    }
    Group& groupRef = objectRef.groups[grpIdx];

    std::vector<Face>& facesRef = groupRef.faces;
    
    if (tokens.size() == 3)
    {
        Face face{};
        face.smoothingGroup = currantSmoothing;
        if (currantMeterial != "")
            face.material = currantMeterial;
        for (int i = 0; i < 3; i++)
        {
            std::vector<std::string> point = split(tokens[i], '/');
            if (point.size() == 1 && !point[0].empty())
                face.vertices[i] = std::stoi(point[0]);
            else if (point.size() == 2 && !point[0].empty() && !point[1].empty())
            {
                face.vertices[i] = std::stoi(point[0]);
                if (!face.textureCoords.has_value())
                    face.textureCoords.emplace();
                (*face.textureCoords)[i] = std::stoi(point[1]);
            }
            else if (point.size() == 3 && !point[0].empty() && !point[2].empty())
            {
                face.vertices[i] = std::stoi(point[0]);
                if (!point[1].empty())
                {
                    if (!face.textureCoords.has_value())
                        face.textureCoords.emplace();
                    (*face.textureCoords)[i] = std::stoi(point[1]);
                }
                if (!face.normals.has_value())
                    face.normals.emplace();
                (*face.normals)[i] = std::stoi(point[2]);
            }
            else
                throw std::runtime_error("OBJ file Corrupted: Face declaration does not follow standarts. at line - " + std::to_string(reading_line));            
        }
        if (!correctFace(face, vertices, textureCoords, normals))
            throw std::runtime_error("OBJ file Corrupted: Face declaration Indexx out of range. at line - " + std::to_string(reading_line));
        facesRef.push_back(std::move(face));
    }
    else
    {
        const int m = static_cast<int>(tokens.size());
        if (m < 3)
            throw std::runtime_error("OBJ face with < 3 vertices" + std::to_string(reading_line));

        std::vector<int> vIdx;                 vIdx.reserve(m);
        std::vector<std::optional<int>> tIdx;  tIdx.reserve(m);
        std::vector<std::optional<int>> nIdx;  nIdx.reserve(m);

        auto parse_one = [&](const std::string& tok)
        {
            auto parts = split(tok, '/');

            if (parts.empty() || parts[0].empty())
                throw std::runtime_error("OBJ face token missing vertex index at line - " + std::to_string(reading_line));

            vIdx.push_back(std::stoi(parts[0]));

            if (parts.size() >= 2 && !parts[1].empty())
                tIdx.push_back(std::stoi(parts[1]));
            else
                tIdx.push_back(std::nullopt);

            if (parts.size() >= 3 && !parts[2].empty())
                nIdx.push_back(std::stoi(parts[2]));
            else
                nIdx.push_back(std::nullopt);
        };

        for (const auto& tok : tokens)
            parse_one(tok);

        std::vector<glm::vec3> poly3D; poly3D.reserve(m);
        for (int i = 0; i < m; ++i)
        {
            int idx = vIdx[i];
            int vi = idx - 1;
            if (vi < 0 || vi >= static_cast<int>(vertices.size()))
                throw std::runtime_error("Vertex index out of range while triangulating at line - " + std::to_string(reading_line));
            poly3D.push_back(vertices[vi]);
        }

        std::vector<glm::vec2> poly2D;
        PlaneBasis basis;
        if (!projectPolygonTo2D(poly3D, poly2D, &basis))
            throw std::runtime_error("Failed to project polygon for ear clipping at line - " + std::to_string(reading_line));

        std::vector<int> mapToOrig(m);
        std::iota(mapToOrig.begin(), mapToOrig.end(), 0);

        auto make_face_from_orig = [&](int aOrig, int bOrig, int cOrig)
        {
            Face face{};
            face.smoothingGroup = currantSmoothing;
            if (!currantMeterial.empty())
                face.material = currantMeterial;

            face.vertices[0] = vIdx[aOrig];
            face.vertices[1] = vIdx[bOrig];
            face.vertices[2] = vIdx[cOrig];

            if (tIdx[aOrig].has_value() || tIdx[bOrig].has_value() || tIdx[cOrig].has_value())
            {
                face.textureCoords.emplace();
                (*face.textureCoords)[0] = tIdx[aOrig].value_or(0);
                (*face.textureCoords)[1] = tIdx[bOrig].value_or(0);
                (*face.textureCoords)[2] = tIdx[cOrig].value_or(0);
            }

            if (nIdx[aOrig].has_value() || nIdx[bOrig].has_value() || nIdx[cOrig].has_value())
            {
                face.normals.emplace();
                (*face.normals)[0] = nIdx[aOrig].value_or(0);
                (*face.normals)[1] = nIdx[bOrig].value_or(0);
                (*face.normals)[2] = nIdx[cOrig].value_or(0);
            }

            if (!correctFace(face, vertices, textureCoords, normals))
                throw std::runtime_error("OBJ file Corrupted: Face declaration Indexx out of range. at line - " + std::to_string(reading_line));
            facesRef.push_back(std::move(face));
        };

        std::vector<glm::vec2> work = poly2D;
        int guard = 0;
        const int maxIter = 5 * m;

        while (static_cast<int>(work.size()) > 3)
        {
            bool clipped = false;
            const int n = static_cast<int>(work.size());

            for (int i = 0; i < n; ++i)
            {
                int prev = (i - 1 + n) % n;
                int next = (i + 1) % n;

                if (!isEar(work, prev, i, next))
                    continue;

                int aOrig = mapToOrig[prev];
                int bOrig = mapToOrig[i];
                int cOrig = mapToOrig[next];

                make_face_from_orig(aOrig, bOrig, cOrig);

                work.erase(work.begin() + i);
                mapToOrig.erase(mapToOrig.begin() + i);

                clipped = true;
                break;
            }

            if (!clipped)
                throw std::runtime_error("Ear clipping failed (no ear found). Check polygon validity at line - " + std::to_string(reading_line));

            if (++guard > maxIter)
                throw std::runtime_error("Ear clipping exceeded iteration guard at line - " + std::to_string(reading_line));
        }

        if (work.size() == 3)
        {
            int aOrig = mapToOrig[0];
            int bOrig = mapToOrig[1];
            int cOrig = mapToOrig[2];
            make_face_from_orig(aOrig, bOrig, cOrig);
        }
    }
}

void Object::loadMaterials()
{
    materials.insert({"", getDefaultMaterial()});
    if (mtlFilePaths.empty())
        return;
    for (const auto& filePath : mtlFilePaths)
    {
        std::ifstream file(filePath);
        if (!file.is_open())
            continue;
        std::string line;
        while (std::getline(file, line))
        {
            int type = -1;
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;

            while (iss >> token)
                tokens.push_back(token);

            if (tokens.empty())
            {
                reading_line++;
                continue;
            }
            if (!tokens[0].empty() && tokens[0][0] == '#')
            {
                reading_line++;
                continue;
            }
            if (tokens[0] == "Kd")
                type = ALBEDO;
            else if (tokens[0] == "Ks")
                type = REFLECTANCE;
            else if (tokens[0] == "Ke")
                type = EMISSION;
            else if (tokens[0] == "Ns")
                type = SHININESS;
            else if (tokens[0] == "d")
                type = OPACITY;
            else if (tokens[0] == "Ni")
                type = REFRACTIVEID;
            else if (tokens[0] == "map_Kd")
                type = TEXTUREFILE;
            else if (tokens[0] == "newmtl")
                type = NAME;
            parseMaterialLine(tokens, type, filePath);
        }
    }
    std::cout << "Loaded Materials" << std::endl;
}

void Object::parseMaterialLine(std::vector<std::string> tokens, int type, std::string filePath)
{
    switch (type)
    {
    case ALBEDO: {
        if (tokens.size() < 4 || currantMTLmat.empty())
            break;
        glm::vec3 albedo{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                break;
            albedo[i] = *f;
        }
        materials[currantMTLmat].albedo = albedo;
        break;
    }
    case REFLECTANCE: {
        if (tokens.size() < 4 || currantMTLmat.empty())
            break;
        glm::vec3 reflectance{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                break;
            reflectance[i] = *f;
        }
        materials[currantMTLmat].reflectance = reflectance;
        break;
    }
    case EMISSION: {
        if (tokens.size() < 4 || currantMTLmat.empty())
            break;
        glm::vec3 emission{};
        for (int i = 0; i < 3; ++i)
        {
            auto f = to_float(tokens[i + 1]);
            if (!f)
                break;
            emission[i] = *f;
        }
        materials[currantMTLmat].emission = emission;
        break;
    }
    case SHININESS: {
        if (tokens.size() < 2 || currantMTLmat.empty())
            break;
        auto f  = to_float(tokens[1]);
        if (!f)
            break;
        materials[currantMTLmat].shininess = *f;
        break;
    }
    case OPACITY: {
        if (tokens.size() < 2 || currantMTLmat.empty())
            break;
        auto f  = to_float(tokens[1]);
        if (!f)
            break;
        materials[currantMTLmat].opacity = *f;
        break;
    }
    case REFRACTIVEID: {
        if (tokens.size() < 2 || currantMTLmat.empty())
            break;
        auto f  = to_float(tokens[1]);
        if (!f)
            break;
        materials[currantMTLmat].refractive_index = *f;
        break;
    }
    case TEXTUREFILE: {
        if (tokens.size() < 2 || currantMTLmat.empty())
            break;
        bool writing_offset = true;
        int index = 0;
        glm::vec3 offset = {0.0f, 0.0f, 0.0f};
        glm::vec3 scale = {1.0f, 1.0f, 1.0f};
        for (int i = 1; i < tokens.size(); i++)
        {
            if (i + 1 == tokens.size())
            {
                filePath = trimAfterLastSlashOrBackslash(filePath);
                materials[currantMTLmat].texture = filePath + tokens[i];
            }
            else if (tokens[i] == "-o")
            {
                index = 0;
                writing_offset = true;
            }
            else if (tokens[i] == "-s")
            {
                index = 0;
                writing_offset = false;
            }
            else if (writing_offset == true)
            {
                auto f = to_float(tokens[i]);
                if (!f)
                    continue;
                offset[index] = *f;
                index++;
            }
            else if (writing_offset == false)
            {
                auto f = to_float(tokens[i]);
                if (!f)
                    continue;
                scale[index] = *f;
                index++;
            }
        }
        materials[currantMTLmat].texture_offset = offset;
        materials[currantMTLmat].texture_scale = scale;
        break;
    }
    case NAME: {
        if (tokens.size() < 2)
        {
            currantMTLmat = "";
            break;
        }
        currantMTLmat = tokens[1];
        materials.insert({currantMTLmat, getDefaultMaterial()});
        break;
    }
    default:
        break;
    }
}

void Object::addNormals()
{
    std::vector<VertexNormalData> vertexData;
    size_t totalFaces = 0;

    for (const auto& object : objects)
        for (const auto& group : object.groups)
            totalFaces += group.faces.size();
    total_faces = totalFaces;
    vertexData.resize((int)vertices.size() + 1);
    for (int vertexID = 0; vertexID < (int)vertices.size(); vertexID++)
        vertexData[vertexID].adjacentNormal.reserve(12);

    for (auto& object : objects)
        for (auto& group : object.groups)
            for (auto& face : group.faces)
            {
                glm::vec3 normal = calculateFaceNormal(vertices[face.vertices[0] - 1], vertices[face.vertices[1] - 1], vertices[face.vertices[2] - 1]);
                for (int n : face.vertices)
                {
                    if (n >= vertices.size() + 1)
                        throw std::runtime_error("Error While calculating normals, Vertex id out of range!");
                    vertexData[n].adjacentNormal.push_back({&face, normal});
                }
            }
    std::cout << "vertex data all found moving on to adding normals!" << std::endl;
    for (int data = 0; data < vertexData.size(); data++)
    {
        std::map<uint32_t, glm::vec3> shadingGroupNormal;
        std::map<uint32_t, int> shadedNormalId;
        for (auto& [face, normal] : vertexData[data].adjacentNormal)
            if (face->smoothingGroup != 0 && glm::length(normal) > EPSILON)
                shadingGroupNormal[face->smoothingGroup] += normal;
        for (auto& [face, normal] : vertexData[data].adjacentNormal)
        {
            
            int i = 0;
            while (i < 3)
            {
                if (face->vertices[i] == data)
                    break;
                i++;
            }
            if (i == 3)
                throw std::runtime_error("Unexpected Exception While generating Normals for a face: " + std::to_string(face->vertices[0]) + " " + std::to_string(face->vertices[1]) + " " + std::to_string(face->vertices[2]));
            if (face->normals.has_value() && (*face->normals)[i] != 0)
                continue;
            if (!face->normals.has_value())
                face->normals.emplace();
            if (face->smoothingGroup == 0)
            {
                if (glm::length(normal) <= EPSILON)
                    normal = glm::vec3{0.0f, 0.0f, 1.0f};
                normals.push_back(glm::normalize(normal));
                int normalID = normals.size();
                for (int j = 0; j < 3; j++)
                    if ((*face->normals)[j] == 0)
                        (*face->normals)[j] = normalID;
            }
            else
            {
                const uint32_t smoothingG = face->smoothingGroup;
                
                glm::vec3 norm;
                if (auto itteratorSmoothingG = shadingGroupNormal.find(smoothingG); itteratorSmoothingG != shadingGroupNormal.end())
                {
                    norm = itteratorSmoothingG->second;
                    if (glm::length(norm) <= EPSILON)
                        norm = glm::vec3(0.0f, 0.0f, 1.0f);
                } 
                else
                    norm = glm::vec3(0.0f, 0.0f, 1.0f);
            
                int normalID = 0;
                if (auto itID = shadedNormalId.find(smoothingG); itID != shadedNormalId.end())
                    normalID = itID->second;
                else
                {
                    normals.push_back(glm::normalize(norm));
                    normalID = static_cast<int>(normals.size());
                    shadedNormalId.emplace(smoothingG, normalID);
                }
            
                (*face->normals)[i] = normalID;
            }
        }
    }
}

std::string Object::getFileName()
{
    std::filesystem::path p(objFilePath);
    return p.stem().string();
}

Material Object::getDefaultMaterial()
{
    Material def{};
    def.name = "";
    def.albedo = glm::vec3{0.8f, 0.8f, 0.8f};
    def.emission = glm::vec3{0.0f, 0.0f, 0.0f};
    def.reflectance = glm::vec3{0.0f, 0.0f, 0.0f};
    def.shininess = 32.0f;
    def.opacity = 1.0f;
    def.refractive_index = 1.0f;
    return def;
}