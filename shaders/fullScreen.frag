// fullScreen.frag
#version 460
layout(location=0) in  vec2 vUV;
layout(location=0) out vec4 outColor;
layout(set=0, binding=0) uniform sampler2D srcLinear;

void main() {
    vec2 uv = vUV;
    vec3 c  = texture(srcLinear, uv).rgb;
    outColor = vec4(c, 1.0);
}