#version 460
layout(push_constant) uniform Push { mat4 uProj; } pc;

layout(location=0) in vec2 inPos;   // unit quad [-0.5..0.5]
layout(location=1) in vec2 iPos;    // top-left px
layout(location=2) in vec2 iSize;   // w,h px
layout(location=3) in vec2 iUvMin;
layout(location=4) in vec2 iUvMax;
layout(location=5) in vec4 iColor;

layout(location=0) out vec2 vUV;
layout(location=1) out vec4 vColor;

void main(){
  vec2 halfSize = 0.5 * iSize;
  vec2 center   = iPos + halfSize;
  vec2 world    = center + inPos * iSize;


  vec2 t  = inPos + vec2(0.5);
  vUV     = mix(iUvMin, iUvMax, t);
  vColor  = iColor;

  gl_Position = pc.uProj * vec4(world, 0.0, 1.0);
}
