#include <stdio.h>
#include <stdlib.h>
#include "viz/shader.h"

GLuint Shader_compile(GLenum type, const char *source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char log[512];
    glGetShaderInfoLog(shader, sizeof(log), NULL, log);
    const char *kind = (type == GL_VERTEX_SHADER) ? "vertex" : "fragment";
    fprintf(stderr, "Shader compile error (%s):\n%s\n", kind, log);
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

GLuint Shader_create_program(const char *vert_source, const char *frag_source) {
  GLuint vert = Shader_compile(GL_VERTEX_SHADER, vert_source);
  if (!vert) return 0;
  GLuint frag = Shader_compile(GL_FRAGMENT_SHADER, frag_source);
  if (!frag) {
    glDeleteShader(vert);
    return 0;
  }

  GLuint program = glCreateProgram();
  glAttachShader(program, vert);
  glAttachShader(program, frag);
  glLinkProgram(program);

  GLint success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char log[512];
    glGetProgramInfoLog(program, sizeof(log), NULL, log);
    fprintf(stderr, "Shader link error:\n%s\n", log);
    glDeleteProgram(program);
    program = 0;
  }

  glDeleteShader(vert);
  glDeleteShader(frag);
  return program;
}

static char *read_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Failed to open shader file: %s\n", path);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc((size_t)len + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t read = fread(buf, 1, (size_t)len, f);
  buf[read] = '\0';
  fclose(f);
  return buf;
}

GLuint Shader_load_program(const char *vert_path, const char *frag_path) {
  char *vert_source = read_file(vert_path);
  if (!vert_source) return 0;
  char *frag_source = read_file(frag_path);
  if (!frag_source) {
    free(vert_source);
    return 0;
  }
  GLuint program = Shader_create_program(vert_source, frag_source);
  free(vert_source);
  free(frag_source);
  return program;
}
