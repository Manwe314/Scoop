#version 460
layout(location=0) in vec2 vUV;
layout(location=0) out vec4 outColor;

layout(binding=0) uniform sampler2D srcLinear;

vec3 aces(vec3 x) {
    const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(srcLinear, vUV).rgb; // linear from compute
    outColor = vec4(hdr, 1.0);              // to *_SRGB swapchain -> HW encodes
}
