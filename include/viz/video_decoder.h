#ifndef VIZ_VIDEO_DECODER_H
#define VIZ_VIDEO_DECODER_H

#include <stdbool.h>
#include <stdint.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

typedef struct {
  AVFormatContext *fmt_ctx;
  AVCodecContext *codec_ctx;
  struct SwsContext *sws_ctx;
  AVFrame *frame;
  AVFrame *frame_rgb;
  AVPacket *pkt;
  int video_stream_idx;
  int width;
  int height;
  uint8_t *rgb_buffer;
  double time_base;     // seconds per PTS tick
  double duration;      // total duration in seconds
  double current_pts;   // PTS of last decoded frame in seconds
} VideoDecoder;

VideoDecoder *VideoDecoder_open(const char *path);
bool VideoDecoder_next_frame(VideoDecoder *dec);
bool VideoDecoder_seek(VideoDecoder *dec, double target_seconds);
void VideoDecoder_close(VideoDecoder *dec);

#endif
