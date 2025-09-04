#version 460

// 1) tiny global state: just the 2D projection matrix (pixels -> clip space)
layout(push_constant) uniform Push {
    mat4 uProj;
} push_constant;

// 2) per-vertex input (Binding 0): the 4 corners of a unit quad in [-0.5, 0.5]^2
layout(location=0) in vec2 inPos;

// 3) per-instance inputs (Binding 1): one record per rectangle
layout(location=1) in vec2 iPos;      // top-left in *pixels*
layout(location=2) in vec2 iSize;     // width & height in pixels
layout(location=3) in vec4 iColor;    // RGBA (0..1)
layout(location=4) in vec4 iRadius;   // corner radii (tl, tr, br, bl), pixels

// 4) varyings we pass to the fragment shader
layout(location=0) out vec2 vLocal;   // pixel coords centered on the rect
layout(location=1) out vec2 vHalf;    // half-size (w/2, h/2)
layout(location=2) out vec4 vColor;   // color
layout(location=3) out vec4 vRadius;  // radii

void main() {
    // 5) derive some handy values
    vec2 halfSize = 0.5 * iSize;      // (w/2, h/2)
    vec2 center   = iPos + halfSize;  // rect center in pixels

    // 6) expand the unit quad to the desired rectangle and place it
    //    inPos is one of: (-0.5,-0.5), (0.5,-0.5), (-0.5,0.5), (0.5,0.5)
    //    multiply by size -> corners relative to the center
    vec2 world = center + inPos * iSize;

    // 7) pass stuff the fragment shader needs for rounded corners, color, etc.
    vLocal  = inPos * iSize;  // local coord with origin at rect center
    vHalf   = halfSize;       // (w/2, h/2)
    vColor  = iColor;
    vRadius = iRadius;

    // 8) final clip-space position
    gl_Position = push_constant.uProj * vec4(world, 0.0, 1.0);
}
