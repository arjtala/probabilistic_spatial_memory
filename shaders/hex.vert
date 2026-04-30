#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec4 a_color;

uniform mat4 u_projection;
uniform vec2 u_extrude_dir;

out vec4 v_color;

void main() {
    vec2 xy = a_position.xy + a_position.z * u_extrude_dir;
    gl_Position = u_projection * vec4(xy, 0.0, 1.0);
    v_color = a_color;
}
