#version 460
layout(set=0, binding=0) uniform sampler2D uAtlas;

layout(location=0) in vec2 vUV;
layout(location=1) in vec4 vColor;
layout(location=0) out vec4 outColor;

void main(){
    // MODE A: visualize UVs
    // outColor = vec4(fract(vUV * 10.0), 0.0, 1.0);

    // MODE B: show atlas region (should look like the glyph bitmap)
    // outColor = vec4(vec3(texture(uAtlas, vUV).r), 1.0);

    // MODE C: just the instance color (prove attribs OK)
    // outColor = vColor;

    // MODE D: your normal premultiplied output
  float a = texture(uAtlas, vUV).r;             // R8_UNORM alpha atlas
  float alpha = vColor.a * a;
  outColor = vec4(vColor.rgb * alpha, alpha);   // premultiplied
}
