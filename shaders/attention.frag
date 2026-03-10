#version 330 core

in vec2 v_texcoord;
out vec4 frag_color;

uniform sampler2D u_attention;
uniform float u_opacity;

// Attempt at Google Turbo colormap approximation
// Maps value in [0,1] to vibrant rainbow: dark blue → cyan → green → yellow → red
vec3 turbo(float t) {
    const vec4 kR = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kG = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kB = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRe = vec2(-152.94239396, 59.28637943);
    const vec2 kGe = vec2(4.27729857, 2.82956604);
    const vec2 kBe = vec2(-89.90310912, 27.34824973);

    t = clamp(t, 0.0, 1.0);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;  // t^4, t^5

    return vec3(
        dot(v4, kR) + dot(v2, kRe),
        dot(v4, kG) + dot(v2, kGe),
        dot(v4, kB) + dot(v2, kBe)
    );
}

void main() {
    float val = texture(u_attention, v_texcoord).r;
    vec3 color = turbo(val);
    float alpha = u_opacity * smoothstep(0.05, 0.3, val);
    frag_color = vec4(color, alpha);
}
