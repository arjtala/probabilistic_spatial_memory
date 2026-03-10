#version 330 core

in vec2 v_texcoord;
out vec4 frag_color;

uniform sampler2D u_attention;
uniform float u_opacity;

// Inferno colormap: black → dark red → orange → yellow → white
// Polynomial approximation (Matt Zucker / matplotlib)
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

void main() {
    float val = texture(u_attention, v_texcoord).r;
    vec3 color = inferno(val);
    float alpha = u_opacity * smoothstep(0.05, 0.3, val);
    frag_color = vec4(color, alpha);
}
