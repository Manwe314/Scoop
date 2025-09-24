#version 460
layout(location=0) out vec2 vUV;

void main() {
    // 3 vertices: (0,1,2)
    const vec2 pos[3] = vec2[3](
        vec2(-1.0, -3.0),
        vec2(-1.0,  1.0),
        vec2( 3.0,  1.0)
    );
    vec2 p = pos[gl_VertexIndex];
    gl_Position = vec4(p, 0.0, 1.0);
    vUV = 0.5 * (p + 1.0);
}
