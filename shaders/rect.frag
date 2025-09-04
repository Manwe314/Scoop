#version 460

// From VS:
layout(location=0) in vec2 vLocal;   // local px coords about rect center
layout(location=1) in vec2 vHalf;    // half size (w/2, h/2)
layout(location=2) in vec4 vColor;   // rgba 0..1
layout(location=3) in vec4 vRadius;  // (tl, tr, br, bl) in pixels

layout(location=0) out vec4 outColor;

// Helper: scale radii if adjacent corners would overlap (CSS-like behavior)
vec4 normalizedRadii(vec2 halfExtent, vec4 r) {
    // clamp to fit within halfExtent extents
    float rmax = min(halfExtent.x, halfExtent.y);
    r = clamp(r, 0.0, rmax);

    // if the sum of two radii along an edge exceeds the edge length, scale all radii down
    float kxTop    = (r.x + r.y) > 0.0 ? min(1.0, halfExtent.x / (r.x + r.y)) : 1.0;
    float kxBottom = (r.w + r.z) > 0.0 ? min(1.0, halfExtent.x / (r.w + r.z)) : 1.0;
    float kyLeft   = (r.x + r.w) > 0.0 ? min(1.0, halfExtent.y / (r.x + r.w)) : 1.0;
    float kyRight  = (r.y + r.z) > 0.0 ? min(1.0, halfExtent.y / (r.y + r.z)) : 1.0;

    float k = min(min(kxTop, kxBottom), min(kyLeft, kyRight));
    return r * k;
}

void main() {
    // normalize radii to be valid for this rect size
    vec4 r = normalizedRadii(vHalf, vRadius);

    // pick the corner radius based on quadrant of the pixel.
    // (remember: with top-left origin, y is negative above center)
    float rc =
        (vLocal.x < 0.0)
          ? ((vLocal.y < 0.0) ? r.x : r.w)   // left side: top-left or bottom-left
          : ((vLocal.y < 0.0) ? r.y : r.z);  // right side: top-right or bottom-right

    // rounded-rect SDF (quadrant-specific radius):
    // see iq's sdRoundBox; here: q = abs(p) - (half - rc)
    vec2  b  = vHalf - vec2(rc);
    vec2  q  = abs(vLocal) - b;
    float sd = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - rc;
    // sd < 0 → inside; sd > 0 → outside; sd = 0 → boundary

    // analytic AA: convert distance to coverage using pixel-size derivatives
    float aa = fwidth(sd);                    // ~ edge softness in pixels
    float coverage = clamp(0.5 - sd / aa, 0.0, 1.0);
    // Equivalently: coverage = 1.0 - smoothstep(0.0, aa, sd);

    // premultiplied output (matches your pipeline blend = ONE, ONE_MINUS_SRC_ALPHA)
    float alpha = vColor.a * coverage;
    outColor = vec4(vColor.rgb * alpha, alpha);
}
