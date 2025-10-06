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

static constexpr int   BINS                 = 16;   //Global
static constexpr float traversalCost        = 1.0f; //Global
static constexpr float intersectionCost     = 1.0f; //Global
static constexpr float Alpha                = 1e-5; //Global

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

static inline bool correctFace(Face& face, const std::vector<glm::vec3>& vertices)
{
    if (face.vertices[0] == 0 || face.vertices[1] == 0 || face.vertices[2] == 0)
        return false;
    if (std::abs(face.vertices[0]) > vertices.size() || std::abs(face.vertices[1]) > vertices.size() || std::abs(face.vertices[2]) > vertices.size())
        return false;
    if (face.textureCoords.has_value())
        if (std::abs((*face.textureCoords)[0]) > vertices.size() || std::abs((*face.textureCoords)[1]) > vertices.size() || std::abs((*face.textureCoords)[2]) > vertices.size())
            return false;
    if (face.normals.has_value())
        if (std::abs((*face.normals)[0]) > vertices.size() || std::abs((*face.normals)[1]) > vertices.size() || std::abs((*face.normals)[2]) > vertices.size())
            return false;
    for (int i = 0; i < 3; i++)
    {
        if (face.vertices[i] < 0)
            face.vertices[i] = vertices.size() - std::abs(face.vertices[i]) + 1;
        if (face.textureCoords.has_value())
            if ((*face.textureCoords)[i] < 0)
                (*face.textureCoords)[i] = vertices.size() - std::abs((*face.textureCoords)[i]) + 1;
        if (face.normals.has_value())
            if ((*face.normals)[i] < 0)
                (*face.normals)[i] = vertices.size() - std::abs((*face.normals)[i]) + 1;
    }
    return true;
}

static void insertOrReplace(std::vector<VertexNormalData>& vec, VertexNormalData data) {
    for (auto& v : vec)
    {
        if (v.id == data.id)
        {
            v = std::move(data);
            return;
        }
    }
    vec.push_back(std::move(data));
}

static inline bool shouldBeLeaf(uint32_t count, int depth, const AABB& centroidBB)
{
    const uint32_t LEAF_THRESHOLD = 4; //Gloabl
    if (count <= LEAF_THRESHOLD)
        return true;
    if (depth  > 64) //Global
        return true;
    
    glm::vec3 d = centroidBB.max - centroidBB.min;
    
    if (d.x <= EPSILON && d.y <= EPSILON && d.z <= EPSILON)
        return true;
    return false;
}

static inline void getFacePositions(const Face* f, const std::vector<glm::vec3>& vertices, glm::vec3& v0, glm::vec3& v1, glm::vec3& v2)
{
    v0 = vertices[ f->vertices[0] - 1 ];
    v1 = vertices[ f->vertices[1] - 1 ];
    v2 = vertices[ f->vertices[2] - 1 ];
}

static inline void computeNodeAndCentroidBounds(const std::vector<TriRef>& refs, uint32_t first, uint32_t count, AABB& nodeBox, AABB& centroidBox)
{
    nodeBox = makeEmptyAABB();
    centroidBox = makeEmptyAABB();
    for (uint32_t i=0; i<count; ++i) {
        const auto& r = refs[first + i];
        nodeBox = merge(nodeBox, r.boundingBox);
        glm::vec3 c = 0.5f * (r.boundingBox.min + r.boundingBox.max);
        expand(centroidBox, c);
    }
}

static bool shouldTrySpatialSplit(const SplitResult& bestObj, float rootSA)
{
    if (!bestObj.valid)
        return false;
    AABB overlap = intersectAABB(bestObj.leftBB, bestObj.rightBB);
    if (!hasPositiveExtent(overlap))
        return false;
    
    float overlapSA = surfaceArea(overlap);
    float fraction = overlapSA / rootSA;
    if (fraction <= Alpha)
        return false;
    return true;
}

static PartitionResult partitionObjectSplit(std::vector<TriRef>& referances, uint32_t first, uint32_t count, int axis, float pos)
{
    uint32_t i = first;
    uint32_t j = first + count - 1;

    AABB leftBB  = makeEmptyAABB();
    AABB rightBB = makeEmptyAABB();

    while (i <= j)
    {
        const AABB& boundingBox = referances[i].boundingBox;
        if (centroidAxis(boundingBox, axis) <= pos)
        {
            mergeInto(leftBB, boundingBox);
            i++;
        }
        else
        {
            std::swap(referances[i], referances[j]);
            mergeInto(rightBB, referances[j].boundingBox);
            if (j == 0) break;
            --j;
        }
    }

    PartitionResult out{};
    out.mid        = i;
    out.leftCount  = i - first;
    out.rightCount = first + count - i;
    out.rightStart = i;
    out.leftBB     = out.leftCount  ? leftBB  : makeEmptyAABB();
    out.rightBB    = out.rightCount ? rightBB : makeEmptyAABB();
    return out;
}

static PartitionResult partitionSpatialWithDup(std::vector<TriRef>& referances, uint32_t first, uint32_t count, int axis, float planePos)
{
    std::vector<TriRef> leftTmp;
    leftTmp.reserve(count);
    std::vector<TriRef> rightTmp;
    rightTmp.reserve(count);

    AABB leftBB  = makeEmptyAABB();
    AABB rightBB = makeEmptyAABB();

    for (uint32_t k = 0; k < count; k++)
    {
        const TriRef& ref = referances[first + k];
        const AABB&   boundingBox = ref.boundingBox;
        float minA = boundingBox.min[axis];
        float maxA = boundingBox.max[axis];

        if (maxA <= planePos) {
            leftTmp.push_back(ref);
            mergeInto(leftBB, boundingBox);
        }
        else if (minA >= planePos) {
            rightTmp.push_back(ref);
            mergeInto(rightBB, boundingBox);
        }
        else {
            TriRef leftRef = ref;
            TriRef rightRef = ref;

            leftRef.boundingBox = clipAABBToHalfspace(boundingBox, axis, planePos, true);
            rightRef.boundingBox = clipAABBToHalfspace(boundingBox, axis, planePos, false);

            if (hasPositiveExtent(leftRef.boundingBox))
            {
                leftTmp.push_back(leftRef);
                mergeInto(leftBB, leftRef.boundingBox);
            }
            if (hasPositiveExtent(rightRef.boundingBox))
            {
                rightTmp.push_back(rightRef);
                mergeInto(rightBB, rightRef.boundingBox);
            }
        }
    }

    const uint32_t L = static_cast<uint32_t>(leftTmp.size());
    if (L > count)
        throw std::runtime_error("While Buildin SBVH: left child larger than parent count");
    for (uint32_t i = 0; i < L; ++i)
        referances[first + i] = leftTmp[i];

    const uint32_t appendBase = static_cast<uint32_t>(referances.size());
    referances.insert(referances.end(), rightTmp.begin(), rightTmp.end());
    const uint32_t R = static_cast<uint32_t>(rightTmp.size());

    PartitionResult out{};
    out.leftCount  = L;
    out.rightCount = R;
    out.rightStart = appendBase;
    out.leftBB     = (L ? leftBB  : makeEmptyAABB());
    out.rightBB    = (R ? rightBB : makeEmptyAABB());
    return out;
}

Object::Object(std::string &filePath) : vertices(), textureCoords(), normals(), mtlFilePaths(), objects(), materials()
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
        if (!correctFace(face, vertices))
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

            if (!correctFace(face, vertices))
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
    vertexData.reserve(totalFaces * 3);

    for (auto& object : objects)
        for (auto& group : object.groups)
            for (auto& face : group.faces)
            {
                for (int n : face.vertices)
                {
                    VertexNormalData vertex = findVertexData(n, vertexData);
                    glm::vec3 normal = calculateFaceNormal(vertices[face.vertices[0] - 1], vertices[face.vertices[1] - 1], vertices[face.vertices[2] - 1]);
                    vertex.adjacentNormal.insert({&face, normal});
                    insertOrReplace(vertexData, vertex);
                }
            }
    for (auto & data : vertexData)
    {
        std::map<uint32_t, glm::vec3> shadingGroupNormal;
        std::map<uint32_t, int> shadedNormalId;
        for (auto& [face, normal] : data.adjacentNormal)
            if (face->smoothingGroup != 0 && glm::length(normal) > EPSILON)
                shadingGroupNormal[face->smoothingGroup] += normal;
        for (auto& [face, normal] : data.adjacentNormal)
        {
            int i = 0;
            while (i < 3)
            {
                if (face->vertices[i] == data.id)
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

std::vector<TriRef> Object::getRefArray() const
{
    std::vector<TriRef> output;
    if (total_faces == 0)
        throw std::runtime_error("Normals creation failed, total faces counted 0");
    output.reserve(total_faces);

    auto getVertex = [&](int index) -> const glm::vec3& {
        int i = index - 1;
        if (i < 0 || i >= vertices.size())
            throw std::runtime_error("Object Parser Failed, Face vertex out of bounds Found");
        return vertices[(size_t)i];
    };

    for (const auto& object : objects)
        for (const auto& group : object.groups)
            for (const auto& face : group.faces)
            {
                const glm::vec3& v0 = getVertex(face.vertices[0]);
                const glm::vec3& v1 = getVertex(face.vertices[1]);
                const glm::vec3& v2 = getVertex(face.vertices[2]);

                const float area = glm::length(glm::cross(v1 - v0, v2 - v0));
                if (area <= EPSILON)
                    continue;
                
                std::vector<glm::vec3> trinagle;
                trinagle.push_back(v0);
                trinagle.push_back(v1);
                trinagle.push_back(v2);
                TriRef refrance{};
                refrance.referance = const_cast<Face*>(&face);
                refrance.boundingBox = getAABB(trinagle);
                
                output.push_back(refrance);
            }
    return output;
}


SBVH Object::buildSplitBoundingVolumeHierarchy()
{
    std::vector<TriRef> referances = getRefArray();
    
    BuildState state{};
    state.refs = std::move(referances);
    state.nodes.reserve(2 * state.refs.size());

    AABB rootBB = getAABB(vertices);
    float rootArea = surfaceArea(rootBB);

    std::vector<BuildTask> stack;
    stack.reserve(2u * state.refs.size());

    BuildTask root{};
    root.first = 0;
    root.count = static_cast<uint32_t>(state.refs.size());
    root.parent = -1;
    root.depth = 0;
    root.asLeft = false;
    root.rootSA = rootArea;

    stack.push_back(root);

    std::vector<uint32_t> leafFirstIndexes;
    std::vector<uint32_t> leafcounts;

    while(!stack.empty())
    {
        BuildTask task = stack.back();
        stack.pop_back();

        int nodeIdx = static_cast<int>(state.nodes.size());
        BVHNode node{};
        state.nodes.push_back(node);

        if (task.parent >= 0)
        {
            if (task.asLeft)
            {
                state.nodes[task.parent].left_start = static_cast<uint32_t>(nodeIdx);
                state.nodes[task.parent].left_count = 0xFFFFFFFFu;
            }
            else
            {
                state.nodes[task.parent].right_start = static_cast<uint32_t>(nodeIdx);
                state.nodes[task.parent].right_count = 0xFFFFFFFFu;
            }
        }

        AABB nodeBB = makeEmptyAABB();
        AABB centroidBB = makeEmptyAABB();
        for (uint32_t i = 0; i < task.count; i++)
        {
            const AABB box = state.refs[task.first + i].boundingBox;
            expand(nodeBB, box.min);
            expand(nodeBB, box.max);
            expand(centroidBB, centroid(box));
        }

        if (shouldBeLeaf(task.count, task.depth, centroidBB))
        {
            state.nodes[nodeIdx].bbMax_left = nodeBB.max;
            state.nodes[nodeIdx].bbMin_left = nodeBB.min;
            state.nodes[nodeIdx].bbMax_right = nodeBB.max;
            state.nodes[nodeIdx].bbMin_right = nodeBB.min;

            state.nodes[nodeIdx].left_start = task.first;
            state.nodes[nodeIdx].left_count = task.count;
            state.nodes[nodeIdx].right_start = 0;
            state.nodes[nodeIdx].right_count = 0;

            leafFirstIndexes.push_back(task.first);
            leafcounts.push_back(task.count);

            continue;
        }

        SplitResult bestObjectSplit = findBestObjectSplit(state.refs, task.first, task.count, centroidBB, nodeBB);

        SplitResult bestSpatialSplit{};
        bestSpatialSplit.valid = false;
        if (shouldTrySpatialSplit(bestObjectSplit, rootArea))
            bestSpatialSplit = findBestSpatialSplit(state.refs, task.first, task.count, centroidBB, nodeBB);
        
        const SplitResult* chosen = nullptr;
        if (!bestObjectSplit.valid && !bestSpatialSplit.valid)
        {
            state.nodes[nodeIdx].bbMin_left  = nodeBB.min;
            state.nodes[nodeIdx].bbMax_left  = nodeBB.max;
            state.nodes[nodeIdx].bbMin_right = nodeBB.min;
            state.nodes[nodeIdx].bbMax_right = nodeBB.max;

            state.nodes[nodeIdx].left_start  = task.first;
            state.nodes[nodeIdx].left_count  = task.count;
            state.nodes[nodeIdx].right_start = 0;
            state.nodes[nodeIdx].right_count = 0;

            leafFirstIndexes.push_back(task.first);
            leafcounts.push_back(task.count);
            continue;
        }
        else if (!bestSpatialSplit.valid)
            chosen = &bestObjectSplit;
        else if (!bestObjectSplit.valid)
            chosen = &bestSpatialSplit;
        else
            chosen = (bestSpatialSplit.cost < bestObjectSplit.cost) ? &bestSpatialSplit : &bestObjectSplit;

        
        PartitionResult partition{};
        uint32_t left_start  = task.first;
        uint32_t right_start = 0;

        if (chosen->spatial)
        {
            partition = partitionSpatialWithDup(state.refs, task.first, task.count, chosen->axis, chosen->pos);
            right_start = partition.rightStart;
        }
        else
        {
            partition = partitionObjectSplit(state.refs, task.first, task.count, chosen->axis, chosen->pos);
            right_start = partition.rightStart;
        }

        if (partition.leftCount == 0 || partition.rightCount == 0)
        {
            state.nodes[nodeIdx].bbMin_left  = nodeBB.min;
            state.nodes[nodeIdx].bbMax_left  = nodeBB.max;
            state.nodes[nodeIdx].bbMin_right = nodeBB.min;
            state.nodes[nodeIdx].bbMax_right = nodeBB.max;

            state.nodes[nodeIdx].left_start  = task.first;
            state.nodes[nodeIdx].left_count  = task.count;
            state.nodes[nodeIdx].right_start = 0;
            state.nodes[nodeIdx].right_count = 0;

            leafFirstIndexes.push_back(task.first);
            leafcounts.push_back(task.count);
            continue;
        }

        state.nodes[nodeIdx].bbMin_left  = partition.leftBB.min;
        state.nodes[nodeIdx].bbMax_left  = partition.leftBB.max;
        state.nodes[nodeIdx].bbMin_right = partition.rightBB.min;
        state.nodes[nodeIdx].bbMax_right = partition.rightBB.max;

        state.nodes[nodeIdx].left_start  = left_start;
        state.nodes[nodeIdx].left_count  = partition.leftCount;
        state.nodes[nodeIdx].right_start = right_start;
        state.nodes[nodeIdx].right_count = partition.rightCount;

        BuildTask left{};
        left.first  = left_start;
        left.count  = partition.leftCount;
        left.parent = nodeIdx;
        left.depth  = task.depth + 1;
        left.asLeft = true;
        left.rootSA = task.rootSA;
        stack.push_back(left);

        BuildTask right{};
        right.first  = right_start;
        right.count  = partition.rightCount;
        right.parent = nodeIdx;
        right.depth  = task.depth + 1;
        right.asLeft = false;
        right.rootSA = task.rootSA;
        stack.push_back(right);
    }
    SBVH splitBoundingVolumeHierarchy{};
    splitBoundingVolumeHierarchy.nodes = std::move(state.nodes);
    splitBoundingVolumeHierarchy.outerBoundingBox = rootBB;
    //build moller Tringales array! + add material maping function
    return splitBoundingVolumeHierarchy;
}

SplitResult Object::findBestObjectSplit(const std::vector<TriRef>& referances, uint32_t first, uint32_t count,const AABB& centroidBB, const AABB& nodeBB)
{
    SplitResult bestSplit{};
    bestSplit.valid = false;
    bestSplit.spatial = false;
    bestSplit.cost = std::numeric_limits<float>::infinity();

    glm::vec3 centroidExtent = centroidBB.max - centroidBB.min;
    if (centroidExtent.x <= EPSILON && centroidExtent.y <= EPSILON && centroidExtent.z <= EPSILON)
        return bestSplit;

    for (int axis = 0; axis < 3; axis++)
    {
        float centroid_min = centroidBB.min[axis];
        float centorid_max = centroidBB.max[axis];
        float extent = centorid_max - centroid_min;
        if (extent <= EPSILON)
            continue;
        
        Bin bins[BINS];
        for (int i = 0; i < BINS; i++)
        {
            bins[i].bb = makeEmptyAABB();
            bins[i].count = 0;
        }

        for (uint32_t i = 0; i < count; i++)
        {
            const AABB& refBB = referances[first + i].boundingBox;
            float center2D = centroidAxis(refBB, axis);
            float rattio = (center2D - centroid_min) / safeExtent(extent);
            int binId = glm::clamp(int(rattio * float(BINS)), 0, BINS - 1);
            mergeInto(bins[binId].bb, refBB);
            bins[binId].count++;
        }

        Bin leftBins[BINS], rightBins[BINS];

        Bin accumulatorLeft;
        accumulatorLeft.bb = makeEmptyAABB();
        accumulatorLeft.count = 0;
        for (int i = 0; i < BINS; i++)
        {
            if (bins[i].count)
                mergeInto(accumulatorLeft.bb, bins[i].bb);
            accumulatorLeft.count += bins[i].count;
            leftBins[i].bb = accumulatorLeft.bb;
            leftBins[i].count = accumulatorLeft.count;
        }

        Bin accumulatorRight;
        accumulatorRight.bb = makeEmptyAABB();
        accumulatorRight.count = 0;
        for (int i = BINS - 1; i >= 0; i--)
        {
            if (bins[i].count)
                mergeInto(accumulatorRight.bb, bins[i].bb);
            accumulatorRight.count += bins[i].count;
            rightBins[i].bb = accumulatorRight.bb;
            rightBins[i].count = accumulatorRight.count;
        }

        float parentSurficeArea = surfaceArea(nodeBB);
        if (parentSurficeArea <= 0.0f)
            continue;
        for (int cut = 0; cut < BINS - 1; cut++)
        {
            uint32_t P_left = leftBins[cut].count;
            uint32_t P_right = rightBins[cut + 1].count;
            if (P_left == 0 || P_right == 0)
                continue;
            float surfaceArea_left = surfaceArea(leftBins[cut].bb);
            float surfaceArea_right = surfaceArea(rightBins[cut + 1].bb);

            float cost = traversalCost + ((surfaceArea_left / parentSurficeArea) * (float)P_left * intersectionCost) + ((surfaceArea_right / parentSurficeArea) * (float)P_right * intersectionCost);

            if (cost < bestSplit.cost)
            {
                bestSplit.valid      = true;
                bestSplit.spatial    = false;
                bestSplit.axis       = axis;
                bestSplit.cost       = cost;
                bestSplit.leftBB     = leftBins[cut].bb;
                bestSplit.rightBB    = rightBins[cut + 1].bb;
                bestSplit.leftCount  = P_left;
                bestSplit.rightCount = P_right;

                float t = float(cut + 1) / float(BINS);
                bestSplit.pos = centroid_min + t * extent;
            }
        }
    }
    return bestSplit;
}

SplitResult Object::findBestSpatialSplit(const std::vector<TriRef>& referances, uint32_t first, uint32_t count, const AABB& centroidBB ,const AABB& nodeBB)
{
    SplitResult bestSplit{};
    bestSplit.valid = false;
    bestSplit.spatial = true;
    bestSplit.cost = std::numeric_limits<float>::infinity();

    if (count == 0)
        return bestSplit;
    float parentSA = surfaceArea(nodeBB);
    if (parentSA <= 0.0f)
        return bestSplit;

    for (int axis = 0; axis < 3; axis++)
    {
        float centroid_min = centroidBB.min[axis];
        float centroid_max = centroidBB.max[axis];
        float extent = centroid_max - centroid_min;
        if (extent <= EPSILON)
            continue;
        float inverseWidth = float(BINS) / safeExtent(extent);

        std::array<std::vector<uint32_t>, BINS> enter;
        std::array<std::vector<uint32_t>, BINS> exit;

        std::array<Bin, BINS> lastBin;
        std::array<Bin, BINS> firstBin;

        for (int i = 0; i < BINS; i++)
        {
            lastBin[i].bb = makeEmptyAABB();
            firstBin[i].bb = makeEmptyAABB();
            lastBin[i].count = 0;
            firstBin[i].count = 0;
        }

        std::vector<int> firstBinIndex(count);
        std::vector<int> lastBinIndex(count);

        for (uint32_t i = 0; i < count; i++)
        {
            const AABB& refBB = referances[first + i].boundingBox;

            float axisMin = refBB.min[axis];
            float axisMax = refBB.max[axis];

            int startBin = glm::clamp(int((axisMin - centroid_min) * inverseWidth), 0, BINS - 1);
            int endBin = glm::clamp(int((axisMax - centroid_min) * inverseWidth), 0, BINS - 1);
            if (endBin < startBin)
                std::swap(endBin, startBin);
            
            enter[startBin].push_back(i);
            exit[endBin].push_back(i);

            firstBinIndex[i] = startBin;
            lastBinIndex[i] = endBin;

            mergeInto(firstBin[startBin].bb, refBB);
            firstBin[startBin].count++;
            
            mergeInto(lastBin[endBin].bb, refBB);
            lastBin[endBin].count++;
        }

        std::array<Bin, BINS> leftFullAtCut;

        Bin accumulatorLeft;
        accumulatorLeft.bb = makeEmptyAABB();
        accumulatorLeft.count = 0;
        for (int i = 0; i < BINS; i++)
        {
            if (lastBin[i].count)
                mergeInto(accumulatorLeft.bb, lastBin[i].bb);
            accumulatorLeft.count += lastBin[i].count;
            leftFullAtCut[i].bb = accumulatorLeft.bb;
            leftFullAtCut[i].count = accumulatorLeft.count;
        }

        std::array<Bin, BINS> rightFullAtCut;

        Bin accumulatorRight;
        accumulatorRight.bb = makeEmptyAABB();
        accumulatorRight.count = 0;
        for (int i = BINS - 1; i >= 0; i--)
        {
            if (firstBin[i].count)
                mergeInto(accumulatorRight.bb, firstBin[i].bb);
            accumulatorRight.count += firstBin[i].count;
            rightFullAtCut[i].bb = accumulatorRight.bb;
            rightFullAtCut[i].count = accumulatorRight.count;
        }

        std::vector<uint32_t> active;
        active.reserve(count);
        std::vector<int> local(count, -1);

        auto addActive = [&](uint32_t localIdx)
        {
            if (local[localIdx] >= 0)
                return;
            local[localIdx] = (int)active.size();
            active.push_back(localIdx);
        };
        auto removeActive = [&](uint32_t localIdx)
        {
            int p = local[localIdx];
            if (p < 0) return;
            uint32_t last = active.back();
            active[p] = last;
            local[last] = p;
            active.pop_back();
            local[localIdx] = -1;
        };
        for (uint32_t i = 0; i < count; ++i)
            if (firstBinIndex[i] < 1 && 1 <= lastBinIndex[i])
                addActive(i);
        for (int k = 1; k < BINS; k++)
        {
            float plane = centroid_min + (float(k) / float(BINS)) * extent;

            for (uint32_t iIdx : exit[k - 1])
                removeActive(iIdx);
            for (uint32_t iIdx: enter[k - 1])
                if (k <= lastBinIndex[iIdx])
                    addActive(iIdx);
            
            Bin leftStradle;
            leftStradle.bb = makeEmptyAABB();
            leftStradle.count = 0;
            
            Bin rightStradle;
            rightStradle.bb = makeEmptyAABB();
            rightStradle.count = 0;

            for (uint32_t localIdx :  active)
            {
                const AABB& ref = referances[first + localIdx].boundingBox;

                AABB leftClippedBB = clipAABBToHalfspace(ref, axis, plane, true);
                if (hasPositiveExtent(leftClippedBB))
                {
                    mergeInto(leftStradle.bb, leftClippedBB);
                    leftStradle.count++;
                }

                AABB rightClippedBB = clipAABBToHalfspace(ref, axis, plane, false);
                if (hasPositiveExtent(rightClippedBB))
                {
                    mergeInto(rightStradle.bb, rightClippedBB);
                    rightStradle.count++;
                }
            }

            Bin childLeft;
            childLeft.bb = mergeBoth(leftFullAtCut[k - 1].bb, leftStradle.bb);
            childLeft.count = leftFullAtCut[k - 1].count + leftStradle.count;
            Bin childright;
            childright.bb = mergeBoth(rightFullAtCut[k].bb, rightStradle.bb);
            childright.count = rightFullAtCut[k].count + rightStradle.count;

            if (childLeft.count == 0 || childright.count == 0)
                continue;
            
            float surfaceArea_left = surfaceArea(childLeft.bb);
            float surfaceArea_right = surfaceArea(childright.bb);

            float cost = traversalCost + ((surfaceArea_left / parentSA) * (float)childLeft.count * intersectionCost) + ((surfaceArea_right / parentSA) * (float)childright.count * intersectionCost);

            if (cost < bestSplit.cost)
            {
                bestSplit.valid      = true;
                bestSplit.spatial    = true;
                bestSplit.axis       = axis;
                bestSplit.pos        = plane;
                bestSplit.cost       = cost;
                bestSplit.leftBB     = childLeft.bb;
                bestSplit.rightBB    = childright.bb;
                bestSplit.leftCount  = childLeft.count;
                bestSplit.rightCount = childright.count;
            }    
        }
    }
    return bestSplit;
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