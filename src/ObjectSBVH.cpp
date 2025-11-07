#include "Object.hpp"

static constexpr int   BINS                 = 16;   //Global
static constexpr float traversalCost        = 1.0f; //Global
static constexpr float intersectionCost     = 1.0f; //Global
static constexpr float Alpha                = 1e-5; //Global

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

static inline bool isLeafNode(BVHNode& node)
{
    if (node.left_count == 0xFFFFFFFFu || node.right_count == 0xFFFFFFFFu)
        return false;
    else
        return true;
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

static inline float shininessToRoughness(float shininess)
{
    shininess = std::max(shininess, 0.0f);
    float roughness = std::sqrt(2.0f / (shininess + 2.0f));
    return glm::clamp(roughness, 0.0f, 1.0f);
}

static inline uint32_t makeFlags(const Material& m) {
    uint32_t f = 0u;
    if (m.opacity >= 0.999f)
        f |= 0x1u;
    return f;
}

static inline MaterialGPU packMaterialGPU(const Material& material, std::vector<std::string>& textures, uint32_t& textureID)
{
    MaterialGPU gpuMaterial{};
    const float rough = shininessToRoughness(material.shininess);
    glm::uvec4 texture{0xFFFFFFFFu};

    gpuMaterial.baseColor_opacity = glm::vec4(material.albedo, glm::clamp(material.opacity, 0.0f, 1.0f));
    gpuMaterial.F0_ior_rough = glm::vec4(material.reflectance, glm::clamp(rough, 0.0f, 1.0f));
    gpuMaterial.emission_flags = glm::vec4(material.emission, glm::uintBitsToFloat(makeFlags(material)));
    if (material.texture.has_value())
    {
        textures.emplace_back(*material.texture);
        texture[0] = textureID;
        textureID++;
    }
    gpuMaterial.textureId = glm::uintBitsToFloat(texture);

    return gpuMaterial;
}

static inline glm::vec4 pack_vec4(const glm::vec3& vec3, uint32_t uint)
{
    return glm::vec4(vec3, glm::uintBitsToFloat(uint));
}

static std::vector<SBVHNode> toSBVH(const std::vector<BVHNode>& src)
{
    std::vector<SBVHNode> out;
    out.reserve(src.size());
    for (const auto& n : src) {
        SBVHNode s;
        s.bbMin_left__left_start   = pack_vec4(n.bbMin_left,  n.left_start);
        s.bbMax_left__left_count   = pack_vec4(n.bbMax_left,  n.left_count);
        s.bbMin_right__right_start = pack_vec4(n.bbMin_right, n.right_start);
        s.bbMax_right__right_count = pack_vec4(n.bbMax_right, n.right_count);
        out.push_back(s);
    }
    return out;
}

uint32_t Object::getMaterialId(const std::optional<std::string>& optName)
{
    if (!optName || optName->empty())
        return 0u;
    auto it = matNameToId.find(*optName);
    if (it != matNameToId.end())
        return it->second;
    uint32_t id = nextMatId++;
    matNameToId.emplace(*optName, id);
    return id;
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

std::vector<MaterialGPU> Object::buildMaterialGPU() 
{
    Material defaultMat = getDefaultMaterial();

    std::vector<MaterialGPU> out(nextMatId + 1, packMaterialGPU(defaultMat, textrues, textureId));

    for (const auto& [name, id] : matNameToId)
    {
        auto it = materials.find(name);
        const Material& src = (it != materials.end()) ? it->second : defaultMat;
        out[id] = packMaterialGPU(src, textrues, textureId);
    }
    return out;
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
        
        BuildTask right{};
        right.first  = right_start;
        right.count  = partition.rightCount;
        right.parent = nodeIdx;
        right.depth  = task.depth + 1;
        right.asLeft = false;
        right.rootSA = task.rootSA;
        stack.push_back(right);

        BuildTask left{};
        left.first  = left_start;
        left.count  = partition.leftCount;
        left.parent = nodeIdx;
        left.depth  = task.depth + 1;
        left.asLeft = true;
        left.rootSA = task.rootSA;
        stack.push_back(left);
    }
    SBVH splitBoundingVolumeHierarchy{};
    splitBoundingVolumeHierarchy.triangles = constructTriangles(state);
    splitBoundingVolumeHierarchy.nodes = toSBVH(state.nodes);
    splitBoundingVolumeHierarchy.outerBoundingBox = rootBB;
    return splitBoundingVolumeHierarchy;
}

Triangles Object::constructTriangles(BuildState& state)
{
    Triangles output;
    uint32_t triangleBuilder = 0;

    std::vector<uint32_t> stack;
    stack.push_back(0u);

    while(!stack.empty())
    {
        BVHNode& node = state.nodes[stack.back()];
        stack.pop_back();

        if (isLeafNode(node))
        {
            const uint32_t first = node.left_start;
            const uint32_t count = node.left_count;
            if (triangleBuilder == 0)
                assert(triangleBuilder == node.left_start);
            if (node.left_start != triangleBuilder)
                node.left_start = triangleBuilder;

            for (uint32_t i = 0; i < count; i++)
            {
                MollerTriangle intersectionTriangle = makeMollerTriangle(state.refs[first + i]);
                ShadingTriangle shadingTrinagle = makeShadingTriangle(state.refs[first + i]);
                output.intersectionTriangles.push_back(intersectionTriangle);
                output.shadingTriangles.push_back(shadingTrinagle);
            }
            triangleBuilder = (uint32_t)output.intersectionTriangles.size();
        }
        else
        {
            if (node.right_count == 0xFFFFFFFFu)
                stack.push_back(node.right_start);
            if (node.left_count == 0xFFFFFFFFu)
                stack.push_back(node.left_start);
        }
    }
    return output;
}

ShadingTriangle Object::makeShadingTriangle(TriRef& ref)
{
    ShadingTriangle output{};
    Face face = *ref.referance;
    const uint32_t matId = getMaterialId(face.material);
    auto getNoraml = [&](int index) -> const glm::vec3& {
        int i = index - 1;
        if (i < 0 || i >= normals.size())
            throw std::runtime_error("Object Parser Failed, Face Normal out of bounds Found");
        return normals[(size_t)i];
    };

    auto getTexture = [&](int index) -> const glm::vec3& {
        int i = index - 1;
        if (i < 0 || i >= textureCoords.size())
            throw std::runtime_error("Object Parser Failed, Face Texture Coordinate out of bounds Found");
        return textureCoords[(size_t)i];
    };

    if (!face.normals.has_value())
        throw std::runtime_error("Cant Construct SBVH a Face without Normals Found");
    glm::vec3 vn0 = getNoraml((*face.normals)[0]);
    glm::vec3 vn1 = getNoraml((*face.normals)[1]);
    glm::vec3 vn2 = getNoraml((*face.normals)[2]);

    glm::vec3 vt0;
    glm::vec3 vt1;
    glm::vec3 vt2;
    if (face.textureCoords.has_value())
    {
        if ((*face.textureCoords)[0] != 0)
            vt0 = getTexture((*face.textureCoords)[0]);
        else
            vt0 = {0.0f, 0.0f, 0.0f};
        if ((*face.textureCoords)[1] != 0)
            vt1 = getTexture((*face.textureCoords)[1]);
        else
            vt1 = {0.0f, 0.0f, 0.0f};
        if ((*face.textureCoords)[2] != 0)
            vt2 = getTexture((*face.textureCoords)[2]);
        else
            vt2 = {0.0f, 0.0f, 0.0f}; 
    }
    else
    {
        vt0 = {0.0f, 0.0f, 0.0f};
        vt1 = {0.0f, 0.0f, 0.0f};
        vt2 = {0.0f, 0.0f, 0.0f};
    }

    output.vertNormal0_uv = glm::vec4(vn0, vt0.x);
    output.vertNormal1_uv = glm::vec4(vn1, vt1.x);
    output.vertNormal2_uv = glm::vec4(vn2, vt2.x);
    output.texture_materialId = glm::vec4(vt0.y, vt1.y, vt2.y, glm::uintBitsToFloat(matId));
    return output;
}

MollerTriangle Object::makeMollerTriangle(TriRef& ref)
{
    MollerTriangle output{};
    Face face = *ref.referance;
    

    auto getVertex = [&](int index) -> const glm::vec3& {
        int i = index - 1;
        if (i < 0 || i >= vertices.size())
            throw std::runtime_error("Object Parser Failed, Face vertex out of bounds Found");
        return vertices[(size_t)i];
    };
    glm::vec3 v0 = getVertex(face.vertices[0]);
    glm::vec3 v1 = getVertex(face.vertices[1]);
    glm::vec3 v2 = getVertex(face.vertices[2]);

    output.vertex_0 = glm::vec4(v0, 0.0f);
    output.edge_vec1 = glm::vec4((v1 - v0), 0.0f);
    output.edge_vec2 = glm::vec4((v2 - v0), 0.0f);
    return output;
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