#pragma once
#include <cmath>
#include <algorithm>
#include <cassert>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

//
// Column-major 4x4 matrix (GLM-like)
// Stored as 4 glm::vec4 columns: x, y, z, w


struct Mat4
{
    glm::vec4 x;
    glm::vec4 y;
    glm::vec4 z;
    glm::vec4 w;

    Mat4() = default;

    explicit Mat4(float diagonal)
    {
        x = glm::vec4(diagonal, 0.0f,    0.0f,    0.0f);
        y = glm::vec4(0.0f,    diagonal, 0.0f,    0.0f);
        z = glm::vec4(0.0f,    0.0f,    diagonal, 0.0f);
        w = glm::vec4(0.0f,    0.0f,    0.0f,    diagonal);
    }

    Mat4(const glm::vec4& c0,
         const glm::vec4& c1,
         const glm::vec4& c2,
         const glm::vec4& c3)
        : x(c0), y(c1), z(c2), w(c3) {}

    static Mat4 identity() { return Mat4(1.0f); }

    inline glm::vec4& operator[](int c)       { return (&x)[c]; }
    inline const glm::vec4& operator[](int c) const { return (&x)[c]; }

    inline float& operator()(int row, int col)
    {
        return (&x)[col][row];
    }

    inline const float& operator()(int row, int col) const
    {
        return (&x)[col][row];
    }

    inline float* data() { return &x.x; }
    inline const float* data() const { return &x.x; }
};
static_assert(sizeof(Mat4) == 4 * sizeof(glm::vec4), "Mat4 must be 4 vec4 columns");

struct Mat3
{
    glm::vec3 x;
    glm::vec3 y;
    glm::vec3 z;

    Mat3() = default;

    explicit Mat3(float diagonal)
    {
        x = glm::vec3(diagonal, 0.0f,    0.0f);
        y = glm::vec3(0.0f,    diagonal, 0.0f);
        z = glm::vec3(0.0f,    0.0f,    diagonal);
    }

    Mat3(const glm::vec3& c0,
         const glm::vec3& c1,
         const glm::vec3& c2)
        : x(c0), y(c1), z(c2) {}

    static Mat3 identity() { return Mat3(1.0f); }

    inline glm::vec3& operator[](int c)       { return (&x)[c]; }
    inline const glm::vec3& operator[](int c) const { return (&x)[c]; }

    inline float& operator()(int row, int col)
    {
        return (&x)[col][row];
    }

    inline const float& operator()(int row, int col) const
    {
        return (&x)[col][row];
    }

    inline float* data() { return &x.x; }
    inline const float* data() const { return &x.x; }
};


// ----- Mat4 arithmetic -----
inline Mat4 operator+(const Mat4& a, const Mat4& b)
{
    return Mat4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline Mat4 operator-(const Mat4& a, const Mat4& b)
{
    return Mat4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline Mat4& operator+=(Mat4& a, const Mat4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

inline Mat4& operator-=(Mat4& a, const Mat4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
    return a;
}

inline Mat4 operator*(const Mat4& a, float s)
{
    return Mat4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline Mat4 operator*(float s, const Mat4& a)
{
    return a * s;
}




// Matrix * vec4  (column vector)
inline glm::vec4 operator*(const Mat4& m, const glm::vec4& v)
{
    return glm::vec4(
        m(0,0)*v.x + m(0,1)*v.y + m(0,2)*v.z + m(0,3)*v.w,
        m(1,0)*v.x + m(1,1)*v.y + m(1,2)*v.z + m(1,3)*v.w,
        m(2,0)*v.x + m(2,1)*v.y + m(2,2)*v.z + m(2,3)*v.w,
        m(3,0)*v.x + m(3,1)*v.y + m(3,2)*v.z + m(3,3)*v.w
    );
}

// Matrix * matrix (column-major, same as glm)
inline Mat4 operator*(const Mat4& A, const Mat4& B)
{
    Mat4 R(0.0f);
    for (int c = 0; c < 4; ++c)
    {
        glm::vec4 bc = B[c];     // column c of B
        R[c] = A * bc;           // transform that column by A
    }
    return R;
}

// Treat vec3 as (x,y,z,1) for positions
inline glm::vec3 operator*(const Mat4& m, const glm::vec3& v)
{
    glm::vec4 r = m * glm::vec4(v, 1.0f);
    return glm::vec3(r) / r.w;
}

// Treat vec2 as (x,y,0,1)
inline glm::vec2 operator*(const Mat4& m, const glm::vec2& v)
{
    glm::vec4 r = m * glm::vec4(v, 0.0f, 1.0f);
    return glm::vec2(r) / r.w;
}

// ----- Mat3 arithmetic -----
inline Mat3 operator+(const Mat3& a, const Mat3& b)
{
    return Mat3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Mat3 operator-(const Mat3& a, const Mat3& b)
{
    return Mat3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Mat3& operator+=(Mat3& a, const Mat3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline Mat3& operator-=(Mat3& a, const Mat3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

inline Mat3 operator*(const Mat3& a, float s)
{
    return Mat3(a.x * s, a.y * s, a.z * s);
}

inline Mat3 operator*(float s, const Mat3& a)
{
    return a * s;
}


// Mat3 * vec3  (column vector)
inline glm::vec3 operator*(const Mat3& m, const glm::vec3& v)
{
    return glm::vec3(
        m(0,0)*v.x + m(0,1)*v.y + m(0,2)*v.z,
        m(1,0)*v.x + m(1,1)*v.y + m(1,2)*v.z,
        m(2,0)*v.x + m(2,1)*v.y + m(2,2)*v.z
    );
}

inline Mat3 operator*(const Mat3& A, const Mat3& B)
{
    Mat3 R(0.0f);
    for (int c = 0; c < 3; ++c)
    {
        glm::vec3 bc = B[c];   // column c of B
        R[c] = A * bc;         // transform that column by A
    }
    return R;
}

inline Mat4 transpose(const Mat4& m)
{
    Mat4 r(0.0f);
    for (int c = 0; c < 4; ++c)
        for (int r0 = 0; r0 < 4; ++r0)
            r(r0, c) = m(c, r0);
    return r;
}

inline Mat3 transpose(const Mat3& m)
{
    Mat3 r(0.0f);
    for (int c = 0; c < 3; ++c)
        for (int r0 = 0; r0 < 3; ++r0)
            r(r0, c) = m(c, r0);
    return r;
}


inline bool inverse(const Mat4& m, Mat4& invOut)
{
    Mat4 a = m;
    invOut = Mat4(1.0f);

    for (int i = 0; i < 4; ++i)
    {
        // Find pivot row
        int pivot = i;
        float maxAbs = std::fabs(a(i, i));
        for (int row = i + 1; row < 4; ++row)
        {
            float v = std::fabs(a(row, i));
            if (v > maxAbs)
            {
                maxAbs = v;
                pivot = row;
            }
        }

        if (maxAbs < 1e-8f)
            return false; // singular

        // Swap rows
        if (pivot != i)
        {
            for (int col = 0; col < 4; ++col)
            {
                std::swap(a(i, col),     a(pivot, col));
                std::swap(invOut(i,col), invOut(pivot,col));
            }
        }

        // Normalize pivot row
        float pv = a(i, i);
        for (int col = 0; col < 4; ++col)
        {
            a(i, col)     /= pv;
            invOut(i,col) /= pv;
        }

        // Eliminate other rows
        for (int row = 0; row < 4; ++row)
        {
            if (row == i) continue;
            float factor = a(row, i);
            for (int col = 0; col < 4; ++col)
            {
                a(row, col)     -= factor * a(i, col);
                invOut(row,col) -= factor * invOut(i,col);
            }
        }
    }

    return true;
}



inline bool inverse(const Mat3& m, Mat3& invOut)
{
    Mat3 a = m;
    invOut = Mat3::identity();

    for (int i = 0; i < 3; ++i)
    {
        int pivot = i;
        float maxAbs = std::fabs(a(i, i));
        for (int row = i + 1; row < 3; ++row)
        {
            float v = std::fabs(a(row, i));
            if (v > maxAbs) { maxAbs = v; pivot = row; }
        }
        if (maxAbs < 1e-8f) return false;

        if (pivot != i)
        {
            for (int col = 0; col < 3; ++col)
            {
                std::swap(a(i, col),      a(pivot, col));
                std::swap(invOut(i, col), invOut(pivot, col));
            }
        }

        float pv = a(i, i);
        for (int col = 0; col < 3; ++col)
        {
            a(i, col)      /= pv;
            invOut(i, col) /= pv;
        }

        for (int row = 0; row < 3; ++row)
        {
            if (row == i) continue;
            float factor = a(row, i);
            if (factor == 0.0f) continue;
            for (int col = 0; col < 3; ++col)
            {
                a(row, col)      -= factor * a(i, col);
                invOut(row, col) -= factor * invOut(i, col);
            }
        }
    }
    return true;
}



inline Mat4 ortho(float left, float right,
                  float bottom, float top,
                  float zNear = -1.0f,
                  float zFar  =  1.0f)
{
    Mat4 m(1.0f); // start from identity

    const float rl = right  - left;
    const float tb = top    - bottom;
    const float fn = zFar   - zNear;

    m(0,0) =  2.0f / rl;
    m(1,1) =  2.0f / tb;
    m(2,2) = -2.0f / fn;
    m(3,3) =  1.0f;

    m(0,3) = -(right  + left)   / rl;
    m(1,3) = -(top    + bottom) / tb;
    m(2,3) = -(zFar   + zNear)  / fn;

    return m;
}
