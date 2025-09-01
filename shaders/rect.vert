#version 460

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2( 0.5, 0.5),
    vec2(-0.5,  0.5)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
}
