#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "viz/gps_trace.h"

#define GPS_TRACE_INITIAL_CAPACITY 1024
// Half-width of the trace ribbon in projected (equirectangular degree) coordinates.
// ~1.5m at equator — thin line appearance. Scales naturally with map zoom.
#define GPS_LINE_HALF_WIDTH 0.000015
// Camera frustum dimensions (multiples of half-width)
#define FRUSTUM_DEPTH 8.0f       // how far the frustum extends forward
#define FRUSTUM_NEAR_HW 1.5f     // near half-width (at camera)
#define FRUSTUM_FAR_HW 5.0f      // far half-width (field of view edge)

GpsTrace *GpsTrace_new(GLuint program) {
  GpsTrace *gt = calloc(1, sizeof(GpsTrace));
  if (!gt) return NULL;

  gt->program = program;
  gt->u_projection = glGetUniformLocation(program, "u_projection");
  gt->count = 0;
  gt->capacity = GPS_TRACE_INITIAL_CAPACITY;
  gt->vertex_count = 0;
  gt->center_lat = 0.0;
  gt->center_lng = 0.0;
  gt->dirty = false;
  gt->last_upload_count = 0;
  gt->last_center_lat = 0.0;
  gt->last_center_lng = 0.0;

  gt->lats = malloc(gt->capacity * sizeof(double));
  gt->lngs = malloc(gt->capacity * sizeof(double));

  // Pre-allocate reusable vertex buffer (ribbon: 2*N + dot: 6 + heading: 3)
  gt->verts_capacity = GPS_TRACE_INITIAL_CAPACITY * 2 + 9;
  gt->verts = malloc(gt->verts_capacity * 6 * sizeof(float));

  // IMU metadata parallel array
  gt->imu_meta = malloc(gt->capacity * sizeof(ImuPointMeta));
  gt->imu_meta_capacity = gt->capacity;
  gt->has_any_imu = false;

  if (!gt->lats || !gt->lngs || !gt->verts || !gt->imu_meta) {
    free(gt->lats);
    free(gt->lngs);
    free(gt->verts);
    free(gt->imu_meta);
    free(gt);
    return NULL;
  }

  glGenVertexArrays(1, &gt->vao);
  glGenBuffers(1, &gt->vbo);

  glBindVertexArray(gt->vao);
  glBindBuffer(GL_ARRAY_BUFFER, gt->vbo);
  glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_DYNAMIC_DRAW);

  // Layout: vec2 position, vec4 color = 6 floats per vertex
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindVertexArray(0);
  return gt;
}

void GpsTrace_push(GpsTrace *gt, double lat, double lng, const ImuPointMeta *imu) {
  if (!gt) return;

  if (gt->count >= gt->capacity) {
    size_t new_cap = gt->capacity * 2;
    double *new_lats = realloc(gt->lats, new_cap * sizeof(double));
    double *new_lngs = realloc(gt->lngs, new_cap * sizeof(double));
    ImuPointMeta *new_imu = realloc(gt->imu_meta, new_cap * sizeof(ImuPointMeta));
    if (!new_lats || !new_lngs || !new_imu) return;
    gt->lats = new_lats;
    gt->lngs = new_lngs;
    gt->imu_meta = new_imu;
    gt->imu_meta_capacity = new_cap;
    gt->capacity = new_cap;
  }

  gt->lats[gt->count] = lat;
  gt->lngs[gt->count] = lng;
  if (imu) {
    gt->imu_meta[gt->count] = *imu;
    gt->has_any_imu = true;
  } else {
    gt->imu_meta[gt->count].has_imu = false;
    gt->imu_meta[gt->count].motion = MOTION_STATIONARY;
    gt->imu_meta[gt->count].heading_rad = 0.0f;
    gt->imu_meta[gt->count].pitch_rad = 0.0f;
  }
  gt->count++;

  // Incremental running average for center
  double n = (double)gt->count;
  gt->center_lat += (lat - gt->center_lat) / n;
  gt->center_lng += (lng - gt->center_lng) / n;

  gt->dirty = true;
}

void GpsTrace_clear(GpsTrace *gt) {
  if (!gt) return;
  gt->count = 0;
  gt->vertex_count = 0;
  gt->center_lat = 0.0;
  gt->center_lng = 0.0;
  gt->dirty = true;
  gt->last_upload_count = 0;
  gt->has_any_imu = false;
}

void GpsTrace_upload(GpsTrace *gt, double proj_center_lat, double proj_center_lng) {
  if (!gt || gt->count == 0) {
    if (gt) gt->vertex_count = 0;
    return;
  }

  // Project relative to the shared map center so all layers align
  double center_lat = proj_center_lat;
  double center_lng = proj_center_lng;

  // Skip upload if nothing changed
  if (!gt->dirty &&
      gt->count == gt->last_upload_count &&
      center_lat == gt->last_center_lat &&
      center_lng == gt->last_center_lng) {
    return;
  }

  // Ribbon: 2 verts per point, tail icon: 6 verts (frustum or dot)
  size_t ribbon_verts = (gt->count >= 2) ? gt->count * 2 : 0;
  size_t total_verts = ribbon_verts + 6;

  // Grow reusable vertex buffer if needed
  if (total_verts > gt->verts_capacity) {
    size_t new_cap = total_verts * 2;
    float *new_verts = realloc(gt->verts, new_cap * 6 * sizeof(float));
    if (!new_verts) return;
    gt->verts = new_verts;
    gt->verts_capacity = new_cap;
  }

  double cos_center = cos(center_lat * M_PI / 180.0);
  float *v = gt->verts;
  size_t vi = 0;
  float hw = (float)GPS_LINE_HALF_WIDTH;

  // Build triangle strip ribbon (left-right vertex pairs)
  if (gt->count >= 2) {
    float prev_nx = 0.0f, prev_ny = 1.0f;  // fallback normal

    for (size_t i = 0; i < gt->count; i++) {
      float xi = (float)((gt->lngs[i] - center_lng) * cos_center);
      float yi = (float)(gt->lats[i] - center_lat);

      // Compute direction vector
      float dx, dy;
      if (i == 0) {
        float xn = (float)((gt->lngs[1] - center_lng) * cos_center);
        float yn = (float)(gt->lats[1] - center_lat);
        dx = xn - xi;
        dy = yn - yi;
      } else if (i == gt->count - 1) {
        float xp = (float)((gt->lngs[i - 1] - center_lng) * cos_center);
        float yp = (float)(gt->lats[i - 1] - center_lat);
        dx = xi - xp;
        dy = yi - yp;
      } else {
        float xn = (float)((gt->lngs[i + 1] - center_lng) * cos_center);
        float yn = (float)(gt->lats[i + 1] - center_lat);
        float xp = (float)((gt->lngs[i - 1] - center_lng) * cos_center);
        float yp = (float)(gt->lats[i - 1] - center_lat);
        dx = xn - xp;
        dy = yn - yp;
      }

      // Perpendicular normal
      float len = sqrtf(dx * dx + dy * dy);
      float nx, ny;
      if (len > 1e-10f) {
        nx = -dy / len;
        ny = dx / len;
        prev_nx = nx;
        prev_ny = ny;
      } else {
        nx = prev_nx;
        ny = prev_ny;
      }

      float alpha = 0.4f + 0.6f * ((float)i / (float)gt->count);

      // Motion-based color lookup
      float cr, cg, cb;
      if (gt->imu_meta[i].has_imu) {
        switch (gt->imu_meta[i].motion) {
        case MOTION_STATIONARY: cr = 0.3f; cg = 0.3f; cb = 1.0f; break;
        case MOTION_WALKING:    cr = 0.0f; cg = 1.0f; cb = 0.4f; break;
        case MOTION_RUNNING:    cr = 1.0f; cg = 0.3f; cb = 0.0f; break;
        default:                cr = 0.0f; cg = 1.0f; cb = 1.0f; break;
        }
      } else {
        cr = 0.0f; cg = 1.0f; cb = 1.0f;  // cyan fallback
      }

      // Left vertex
      v[vi * 6 + 0] = xi + nx * hw;
      v[vi * 6 + 1] = yi + ny * hw;
      v[vi * 6 + 2] = cr;
      v[vi * 6 + 3] = cg;
      v[vi * 6 + 4] = cb;
      v[vi * 6 + 5] = alpha;
      vi++;

      // Right vertex
      v[vi * 6 + 0] = xi - nx * hw;
      v[vi * 6 + 1] = yi - ny * hw;
      v[vi * 6 + 2] = cr;
      v[vi * 6 + 3] = cg;
      v[vi * 6 + 4] = cb;
      v[vi * 6 + 5] = alpha;
      vi++;
    }
  }

  // Current position indicator (6 verts)
  size_t last = gt->count - 1;
  float cx = (float)((gt->lngs[last] - center_lng) * cos_center);
  float cy = (float)(gt->lats[last] - center_lat);

  bool has_heading = gt->has_any_imu && gt->imu_meta[last].has_imu;

  if (has_heading) {
    // Camera frustum: trapezoid showing field of view in heading direction
    float heading = gt->imu_meta[last].heading_rad;
    float pitch_rad = gt->imu_meta[last].pitch_rad;
    float fwd_x = sinf(heading);
    float fwd_y = cosf(heading);
    float perp_x = cosf(heading);
    float perp_y = -sinf(heading);

    // pitch_factor: 1.0 horizontal, 0.0 straight down
    float pitch_factor = cosf(pitch_rad);
    if (pitch_factor < 0.0f) pitch_factor = 0.0f;
    if (pitch_factor > 1.0f) pitch_factor = 1.0f;

    // Looking forward → long, narrow; looking down → short, wide
    float depth = hw * (3.0f + pitch_factor * 10.0f);
    float far_hw_val = hw * (2.0f + (1.0f - pitch_factor) * 6.0f);
    float near_hw_val = hw * FRUSTUM_NEAR_HW;

    // Near edge (at camera position)
    float nl_x = cx - perp_x * near_hw_val;
    float nl_y = cy - perp_y * near_hw_val;
    float nr_x = cx + perp_x * near_hw_val;
    float nr_y = cy + perp_y * near_hw_val;
    // Far edge (in viewing direction)
    float fl_x = cx + fwd_x * depth - perp_x * far_hw_val;
    float fl_y = cy + fwd_y * depth - perp_y * far_hw_val;
    float fr_x = cx + fwd_x * depth + perp_x * far_hw_val;
    float fr_y = cy + fwd_y * depth + perp_y * far_hw_val;

    float cr = 1.0f, cg = 1.0f, cb = 1.0f, ca = 0.85f;  // white, slightly transparent

    // Triangle 1: near-left, near-right, far-left
    v[vi*6+0] = nl_x; v[vi*6+1] = nl_y;
    v[vi*6+2] = cr; v[vi*6+3] = cg; v[vi*6+4] = cb; v[vi*6+5] = ca;
    vi++;
    v[vi*6+0] = nr_x; v[vi*6+1] = nr_y;
    v[vi*6+2] = cr; v[vi*6+3] = cg; v[vi*6+4] = cb; v[vi*6+5] = ca;
    vi++;
    v[vi*6+0] = fl_x; v[vi*6+1] = fl_y;
    v[vi*6+2] = cr; v[vi*6+3] = cg; v[vi*6+4] = cb; v[vi*6+5] = ca;
    vi++;
    // Triangle 2: near-right, far-right, far-left
    v[vi*6+0] = nr_x; v[vi*6+1] = nr_y;
    v[vi*6+2] = cr; v[vi*6+3] = cg; v[vi*6+4] = cb; v[vi*6+5] = ca;
    vi++;
    v[vi*6+0] = fr_x; v[vi*6+1] = fr_y;
    v[vi*6+2] = cr; v[vi*6+3] = cg; v[vi*6+4] = cb; v[vi*6+5] = ca;
    vi++;
    v[vi*6+0] = fl_x; v[vi*6+1] = fl_y;
    v[vi*6+2] = cr; v[vi*6+3] = cg; v[vi*6+4] = cb; v[vi*6+5] = ca;
    vi++;
  } else {
    // No IMU: magenta dot (square quad)
    float dot_r = hw * 3.0f;
    float dr = 1.0f, dg = 0.0f, db = 1.0f, da = 1.0f;

    v[vi*6+0] = cx - dot_r; v[vi*6+1] = cy - dot_r;
    v[vi*6+2] = dr; v[vi*6+3] = dg; v[vi*6+4] = db; v[vi*6+5] = da;
    vi++;
    v[vi*6+0] = cx + dot_r; v[vi*6+1] = cy - dot_r;
    v[vi*6+2] = dr; v[vi*6+3] = dg; v[vi*6+4] = db; v[vi*6+5] = da;
    vi++;
    v[vi*6+0] = cx - dot_r; v[vi*6+1] = cy + dot_r;
    v[vi*6+2] = dr; v[vi*6+3] = dg; v[vi*6+4] = db; v[vi*6+5] = da;
    vi++;
    v[vi*6+0] = cx - dot_r; v[vi*6+1] = cy + dot_r;
    v[vi*6+2] = dr; v[vi*6+3] = dg; v[vi*6+4] = db; v[vi*6+5] = da;
    vi++;
    v[vi*6+0] = cx + dot_r; v[vi*6+1] = cy - dot_r;
    v[vi*6+2] = dr; v[vi*6+3] = dg; v[vi*6+4] = db; v[vi*6+5] = da;
    vi++;
    v[vi*6+0] = cx + dot_r; v[vi*6+1] = cy + dot_r;
    v[vi*6+2] = dr; v[vi*6+3] = dg; v[vi*6+4] = db; v[vi*6+5] = da;
    vi++;
  }

  gt->vertex_count = vi;

  glBindBuffer(GL_ARRAY_BUFFER, gt->vbo);
  glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vi * 6 * sizeof(float)),
               gt->verts, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Update tracking state
  gt->dirty = false;
  gt->last_upload_count = gt->count;
  gt->last_center_lat = center_lat;
  gt->last_center_lng = center_lng;
}

void GpsTrace_draw(GpsTrace *gt, int viewport_w, int viewport_h, double zoom) {
  if (!gt || gt->vertex_count < 6) return;  // need at least position icon

  glUseProgram(gt->program);

  // Same ortho projection as HexRenderer
  double aspect = (double)viewport_h / (double)viewport_w;
  double half_w = zoom;
  double half_h = zoom * aspect;

  float proj[16] = {0};
  proj[0] = (float)(1.0 / half_w);
  proj[5] = (float)(1.0 / half_h);
  proj[10] = -1.0f;
  proj[15] = 1.0f;

  glUniformMatrix4fv(gt->u_projection, 1, GL_FALSE, proj);

  glBindVertexArray(gt->vao);

  // Draw ribbon (triangle strip) — all vertices except the last 6 (position icon)
  size_t ribbon_count = gt->vertex_count - 6;
  if (ribbon_count >= 4) {
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)ribbon_count);
  }

  // Draw position icon: camera frustum (IMU) or dot (no IMU) — last 6 vertices
  glDrawArrays(GL_TRIANGLES, (GLint)ribbon_count, 6);

  glBindVertexArray(0);
}

void GpsTrace_free(GpsTrace *gt) {
  if (!gt) return;
  glDeleteBuffers(1, &gt->vbo);
  glDeleteVertexArrays(1, &gt->vao);
  free(gt->lats);
  free(gt->lngs);
  free(gt->verts);
  free(gt->imu_meta);
  free(gt);
}
