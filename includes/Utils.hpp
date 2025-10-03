#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <glm/glm.hpp>

#define EPSILON 1e-12

struct PlaneBasis {
    glm::vec3 origin;
    glm::vec3 u;
    glm::vec3 v;
    glm::vec3 n;
};

struct CoplanarityStats {
    double max_abs_dist;
    double mean_abs_dist;
    double rms_dist;
    size_t count_over_tol;
};

struct AABB{
    glm::vec3 min;
    glm::vec3 max;
};

inline glm::vec3 sub3(const glm::vec3& a, const glm::vec3& b)
{
    return {a.x-b.x, a.y-b.y, a.z-b.z};
}

inline double dot3(const glm::vec3& a, const glm::vec3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline glm::vec3 cross3(const glm::vec3& a, const glm::vec3& b)
{
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

inline double norm3(const glm::vec3& a)
{
    return std::sqrt(dot3(a,a));
}

inline glm::vec3 normalize3(const glm::vec3& v)
{
    double n = norm3(v);
    if (n < EPSILON)
        return {0,0,0};
    return { v.x/n, v.y/n, v.z/n };
}

inline double cross2(const glm::vec2& a, const glm::vec2& b)
{
    return a.x*b.y - a.y*b.x;
}

inline glm::vec2 sub2(const glm::vec2& a, const glm::vec2& b)
{
    return {a.x-b.x, a.y-b.y};
}

inline bool buildPlaneBasis(const std::vector<glm::vec3>& P, PlaneBasis& B)
{
    const int n = (int)P.size();
    if (n < 3)
        return false;

    glm::vec3 o = P[0];
    glm::vec3 u = {0,0,0};
    glm::vec3 nrm = {0,0,0};

    for (int i = 1; i < n; ++i)
    {
        glm::vec3 a = sub3(P[i], o);
        if (norm3(a) > EPSILON)
        {
            u = a;
            break;
        }
    }
    if (norm3(u) < EPSILON)
        return false;

    for (int j = 2; j < n; ++j)
    {
        glm::vec3 b = sub3(P[j], o);
        glm::vec3 c = cross3(u, b);
        if (norm3(c) > EPSILON)
        {
            nrm = normalize3(c);
            break;
        }
    }
    if (norm3(nrm) < EPSILON)
        return false;

    glm::vec3 uu = normalize3(u);
    glm::vec3 vv = cross3(nrm, uu);
    B.origin = o;
    B.u = uu;
    B.v = vv;
    B.n = nrm;
    return true;
}

inline glm::vec2 projectTo2D(const glm::vec3& p, const PlaneBasis& B)
{
    glm::vec3 d = sub3(p, B.origin);
    return { dot3(d, B.u), dot3(d, B.v) };
}

inline double polygonArea2D(const std::vector<glm::vec2>& poly)
{
    const int n = (int)poly.size();
    double A = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const glm::vec2& a = poly[i];
        const glm::vec2& b = poly[(i+1)%n];
        A += cross2(a, b);
    }
    return 0.5 * A;
}

inline bool projectPolygonTo2D(const std::vector<glm::vec3>& pts3D, std::vector<glm::vec2>& out2D, PlaneBasis* outBasis = nullptr, bool enforceCCW = true)
{
    out2D.clear();
    if (pts3D.size() < 3)
        return false;

    PlaneBasis B;
    if (!buildPlaneBasis(pts3D, B))
        return false;

    out2D.reserve(pts3D.size());
    for (const glm::vec3& p : pts3D)
        out2D.push_back(projectTo2D(p, B));

    if (enforceCCW)
    {
        double A = polygonArea2D(out2D);
        if (A < -EPSILON)
            std::reverse(out2D.begin(), out2D.end());
    }

    if (outBasis)
        *outBasis = B;
    return true;
}

// evaluate coplain

inline double pointPlaneSignedDistance(const glm::vec3& p, const PlaneBasis& B)
{
    glm::vec3 d = sub3(p, B.origin);
    return dot3(d, B.n); 
}


inline glm::vec3 liftFrom2D(const glm::vec2& q, const PlaneBasis& B)
{
    return { B.origin.x + B.u.x*q.x + B.v.x*q.y, B.origin.y + B.u.y*q.x + B.v.y*q.y, B.origin.z + B.u.z*q.x + B.v.z*q.y };
}

inline double pointReprojectionError(const glm::vec3& p, const PlaneBasis& B)
{
    glm::vec3 d = sub3(p, B.origin);

    double x = dot3(d, B.u);
    double y = dot3(d, B.v);

    glm::vec3 ph = liftFrom2D({x,y}, B);
    glm::vec3 r = sub3(p, ph);
    return norm3(r);
}

inline CoplanarityStats evaluateCoplanarity(const std::vector<glm::vec3>& pts3D, const PlaneBasis& B, double tolerance)
{
    CoplanarityStats S{0.0, 0.0, 0.0, 0};
    if (pts3D.empty())
        return S;

    double sum_abs = 0.0, sum_sq = 0.0, max_abs = 0.0;
    size_t over = 0;

    for (const auto& p : pts3D)
    {
        double d = pointPlaneSignedDistance(p, B);
        double a = std::fabs(d);
        sum_abs += a;
        sum_sq  += d*d;
        if (a > max_abs)
            max_abs = a;
        if (a > tolerance)
            over++;
    }

    const double n = (double)pts3D.size();
    S.max_abs_dist   = max_abs;
    S.mean_abs_dist  = sum_abs / n;
    S.rms_dist       = std::sqrt(sum_sq / n);
    S.count_over_tol = over;
    return S;
}

// is ear


inline double orient2d(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
{
    return cross2(b - a, c - a);
}

inline double signedArea(const std::vector<glm::vec2>& poly)
{
    double A = 0.0;
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i)
    {
        const glm::vec2& p = poly[i];
        const glm::vec2& q = poly[(i+1)%n];
        A += cross2(p, q);
    }
    return 0.5 * A;
}

inline bool pointInTriInclusive(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
{
    const double o1 = orient2d(a, b, p);
    const double o2 = orient2d(b, c, p);
    const double o3 = orient2d(c, a, p);

    const bool nonNeg = (o1 >= -EPSILON) && (o2 >= -EPSILON) && (o3 >= -EPSILON);
    const bool nonPos = (o1 <=  EPSILON) && (o2 <=  EPSILON) && (o3 <=  EPSILON);
    return nonNeg || nonPos;
}

inline bool isEar(const std::vector<glm::vec2>& poly, int prev, int v, int next)
{
    const int n = static_cast<int>(poly.size());
    if (n < 3) return false;

    const glm::vec2& A = poly[prev];
    const glm::vec2& B = poly[v];
    const glm::vec2& C = poly[next];

    const double polyArea = signedArea(poly);
    const bool ccw = (polyArea > 0.0);

    const double turn = orient2d(A, B, C);
    const bool convex = ccw ? (turn > -EPSILON) : (turn < EPSILON);
    if (!convex)
        return false;

    for (int i = 0; i < n; ++i)
    {
        if (i == prev || i == v || i == next)
            continue;
        const glm::vec2& P = poly[i];
        if (pointInTriInclusive(P, A, B, C))
            return false;
    }
    return true;
}

// AABB stuff

inline AABB getAABB(std::vector<glm::vec3>& vertices)
{
    glm::vec3 boxMin = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    glm::vec3 boxMax = {-boxMin.x, -boxMin.y, -boxMin.z};
    
    for (const auto& vertex : vertices)
    {
        boxMin.x = std::min(boxMin.x, vertex.x);
        boxMin.y = std::min(boxMin.y, vertex.y);
        boxMin.z = std::min(boxMin.z, vertex.z);
        
        boxMax.x = std::max(boxMax.x, vertex.x);
        boxMax.y = std::max(boxMax.y, vertex.y);
        boxMax.z = std::max(boxMax.z, vertex.z);
    }
    AABB boundingBox{boxMin, boxMax};
    return boundingBox;
}

inline AABB makeEmptyAABB()
{
    AABB box{};
    box.min = glm::vec3( std::numeric_limits<float>::max());
    box.max = glm::vec3(-std::numeric_limits<float>::max());
    return box;
}

inline void expand(AABB& a, const glm::vec3& b)
{
    a.min = glm::min(a.min, b);
    a.max = glm::max(a.max, b);
}

inline AABB merge(const AABB& a, const AABB& b)
{
    AABB box{};
    box.min = glm::min(a.min, b.min);
    box.max = glm::max(a.max, b.max);
    return box;
}

inline float surfaceArea(const AABB& box)
{
    glm::vec3 dimensions = glm::max(box.max - box.min, glm::vec3(0));
    return 2.0f * (dimensions.x*dimensions.y + dimensions.y*dimensions.z + dimensions.z*dimensions.x);
}

inline void getFacePositions(const Face* f, const std::vector<glm::vec3>& vertices, glm::vec3& v0, glm::vec3& v1, glm::vec3& v2)
{
    v0 = vertices[ f->vertices[0] - 1 ];
    v1 = vertices[ f->vertices[1] - 1 ];
    v2 = vertices[ f->vertices[2] - 1 ];
}

inline void computeNodeAndCentroidBounds(const std::vector<TriRef>& refs, uint32_t first, uint32_t count, AABB& nodeBox, AABB& centroidBox)
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

inline bool centroidDegenerate(const AABB& cbox)
{
    glm::vec3 d = glm::max(cbox.max - cbox.min, glm::vec3(0));
    return (d.x < EPSILON && d.y < EPSILON && d.z < EPSILON);
}

inline glm::vec3 centroid(const AABB& box)
{
    return 0.5f * (box.min + box.max);
}

inline glm::vec3 calculateFaceNormal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    glm::vec3 normal = glm::cross(v1 - v0, v2 - v0);
    return normal;
}
