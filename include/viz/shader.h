#ifndef VIZ_SHADER_H
#define VIZ_SHADER_H

#include <OpenGL/gl3.h>

GLuint Shader_compile(GLenum type, const char *source);
GLuint Shader_create_program(const char *vert_source, const char *frag_source);
GLuint Shader_load_program(const char *vert_path, const char *frag_path);

#endif
