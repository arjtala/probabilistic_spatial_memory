#ifndef VIZ_TILE_PIPELINE_H
#define VIZ_TILE_PIPELINE_H

#include <stdbool.h>
#include <stdint.h>
#include "viz/tile_map.h"

typedef struct {
  uint8_t *pixels;
  int width;
  int height;
  int x;
  int y;
  int z;
} TileDecodedImage;

bool TilePipeline_init(TileMap *tm);
void TilePipeline_shutdown(TileMap *tm);
void TilePipeline_start_download(TileMap *tm, int x, int y, int z);
void TilePipeline_poll_downloads(TileMap *tm);
bool TilePipeline_take_decoded(TileMap *tm, TileDecodedImage *out_image);
void TilePipeline_accumulate_stats(TileMap *tm, TileMapStats *out_stats);

#endif
