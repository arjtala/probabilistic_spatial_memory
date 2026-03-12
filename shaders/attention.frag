#version 330 core

in vec2 v_texcoord;
out vec4 frag_color;

uniform sampler2D u_attention;
uniform float u_opacity;
uniform int u_colormap;  // 0 = inferno (DINO attention), 1 = viridis (JEPA prediction error)

// Inferno: black → dark red → orange → yellow → white
// Warm tones — reads as "intensity of attention"
vec3 inferno(float t) {
    const vec3 c0 = vec3(0.0002, 0.0017, -0.0195);
    const vec3 c1 = vec3(0.1065, 0.5640, 3.9327);
    const vec3 c2 = vec3(11.6025, -3.9729, -15.9424);
    const vec3 c3 = vec3(-41.7040, 17.4364, 44.3541);
    const vec3 c4 = vec3(77.1629, -33.4024, -81.8073);
    const vec3 c5 = vec3(-73.7688, 32.6261, 73.2095);
    const vec3 c6 = vec3(27.1644, -12.2427, -23.0703);

    t = clamp(t, 0.0, 1.0);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), 0.0, 1.0);
}

// Viridis: dark purple → teal → green → yellow
// Cool tones — reads as "prediction error / surprise"
vec3 viridis(float t) {
    const vec3 c0 = vec3(0.2777, 0.0054, 0.3340);
    const vec3 c1 = vec3(0.1050, 1.4046, 1.3845);
    const vec3 c2 = vec3(-0.3308, -0.2149, -19.4525);
    const vec3 c3 = vec3(-4.6342, -5.7991, 56.6905);
    const vec3 c4 = vec3(6.2282, 14.1800, -66.1499);
    const vec3 c5 = vec3(4.7763, -13.7451, 35.1636);
    const vec3 c6 = vec3(-5.4354, 4.6459, -6.9272);

    t = clamp(t, 0.0, 1.0);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), 0.0, 1.0);
}

void main() {
    float val = texture(u_attention, v_texcoord).r;
    // JEPA: flip so low prediction accuracy (surprise) maps to bright yellow
    float mapped = (u_colormap == 1) ? 1.0 - val : val;
    vec3 color = (u_colormap == 1) ? viridis(mapped) : inferno(mapped);
    float alpha = u_opacity * smoothstep(0.05, 0.3, mapped);
    frag_color = vec4(color, alpha);
}
