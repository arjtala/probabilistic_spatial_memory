#include <stdio.h>
#include <stdlib.h>
#include "viz/video_decoder.h"

static void update_current_pts(VideoDecoder *dec) {
  int64_t best_effort_pts = dec->frame->best_effort_timestamp;
  if (best_effort_pts != AV_NOPTS_VALUE) {
    dec->current_pts = (double)best_effort_pts * dec->time_base;
  } else if (dec->frame->pts != AV_NOPTS_VALUE) {
    dec->current_pts = (double)dec->frame->pts * dec->time_base;
  }
}

static void cleanup_decoder(VideoDecoder *dec) {
  if (!dec) return;
  av_free(dec->rgb_buffer);
  av_frame_free(&dec->frame);
  av_frame_free(&dec->frame_rgb);
  av_packet_free(&dec->pkt);
  sws_freeContext(dec->sws_ctx);
  avcodec_free_context(&dec->codec_ctx);
  avformat_close_input(&dec->fmt_ctx);
  free(dec);
}

VideoDecoder *VideoDecoder_open(const char *path) {
  VideoDecoder *dec = calloc(1, sizeof(VideoDecoder));
  if (!dec) return NULL;

  if (avformat_open_input(&dec->fmt_ctx, path, NULL, NULL) < 0) {
    fprintf(stderr, "Failed to open video: %s\n", path);
    free(dec);
    return NULL;
  }

  if (avformat_find_stream_info(dec->fmt_ctx, NULL) < 0) {
    fprintf(stderr, "Failed to find stream info\n");
    avformat_close_input(&dec->fmt_ctx);
    free(dec);
    return NULL;
  }

  // Find best video stream
  dec->video_stream_idx = -1;
  for (unsigned i = 0; i < dec->fmt_ctx->nb_streams; i++) {
    if (dec->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      dec->video_stream_idx = (int)i;
      break;
    }
  }
  if (dec->video_stream_idx < 0) {
    fprintf(stderr, "No video stream found\n");
    avformat_close_input(&dec->fmt_ctx);
    free(dec);
    return NULL;
  }

  AVStream *stream = dec->fmt_ctx->streams[dec->video_stream_idx];
  const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
  if (!codec) {
    fprintf(stderr, "Unsupported codec\n");
    avformat_close_input(&dec->fmt_ctx);
    free(dec);
    return NULL;
  }

  dec->codec_ctx = avcodec_alloc_context3(codec);
  if (!dec->codec_ctx) {
    avformat_close_input(&dec->fmt_ctx);
    free(dec);
    return NULL;
  }
  avcodec_parameters_to_context(dec->codec_ctx, stream->codecpar);
  if (avcodec_open2(dec->codec_ctx, codec, NULL) < 0) {
    fprintf(stderr, "Failed to open codec\n");
    cleanup_decoder(dec);
    return NULL;
  }

  dec->width = dec->codec_ctx->width;
  dec->height = dec->codec_ctx->height;
  dec->time_base = av_q2d(stream->time_base);
  if (stream->duration > 0) {
    dec->duration = (double)stream->duration * dec->time_base;
  } else if (dec->fmt_ctx->duration > 0) {
    dec->duration = (double)dec->fmt_ctx->duration / (double)AV_TIME_BASE;
  }

  // Set up sws scaler to RGB24
  dec->sws_ctx = sws_getContext(dec->width, dec->height, dec->codec_ctx->pix_fmt,
                                dec->width, dec->height, AV_PIX_FMT_RGB24,
                                SWS_BILINEAR, NULL, NULL, NULL);
  if (!dec->sws_ctx) {
    cleanup_decoder(dec);
    return NULL;
  }

  dec->frame = av_frame_alloc();
  dec->frame_rgb = av_frame_alloc();
  dec->pkt = av_packet_alloc();
  if (!dec->frame || !dec->frame_rgb || !dec->pkt) {
    cleanup_decoder(dec);
    return NULL;
  }

  // Allocate RGB buffer
  int rgb_size = dec->width * dec->height * 3;
  dec->rgb_buffer = (uint8_t *)av_malloc((size_t)rgb_size);
  if (!dec->rgb_buffer) {
    cleanup_decoder(dec);
    return NULL;
  }
  dec->frame_rgb->data[0] = dec->rgb_buffer;
  dec->frame_rgb->linesize[0] = dec->width * 3;

  dec->current_pts = 0.0;
  dec->draining = false;
  return dec;
}

bool VideoDecoder_next_frame(VideoDecoder *dec) {
  while (true) {
    int ret = avcodec_receive_frame(dec->codec_ctx, dec->frame);
    if (ret == 0) {
      sws_scale(dec->sws_ctx, (const uint8_t *const *)dec->frame->data,
                dec->frame->linesize, 0, dec->height,
                dec->frame_rgb->data, dec->frame_rgb->linesize);
      update_current_pts(dec);
      return true;
    }
    if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
      return false;
    }
    if (ret == AVERROR_EOF) {
      return false;
    }

    if (dec->draining) {
      int drain_ret = avcodec_send_packet(dec->codec_ctx, NULL);
      if (drain_ret < 0 && drain_ret != AVERROR_EOF) {
        return false;
      }
      dec->draining = false;
      continue;
    }

    bool sent_packet = false;
    while (true) {
      int read_ret = av_read_frame(dec->fmt_ctx, dec->pkt);
      if (read_ret < 0) {
        dec->draining = true;
        break;
      }
      if (dec->pkt->stream_index != dec->video_stream_idx) {
        av_packet_unref(dec->pkt);
        continue;
      }

      ret = avcodec_send_packet(dec->codec_ctx, dec->pkt);
      av_packet_unref(dec->pkt);
      if (ret == AVERROR(EAGAIN)) {
        break;
      }
      if (ret < 0) {
        return false;
      }
      sent_packet = true;
      break;
    }

    if (ret == AVERROR(EAGAIN)) {
      continue;
    }
    if (dec->draining || sent_packet) {
      continue;
    }
  }
}

bool VideoDecoder_seek(VideoDecoder *dec, double target_seconds) {
  if (target_seconds < 0.0) target_seconds = 0.0;
  if (target_seconds > dec->duration) target_seconds = dec->duration;

  int64_t target_ts = (int64_t)(target_seconds / dec->time_base);

  if (av_seek_frame(dec->fmt_ctx, dec->video_stream_idx, target_ts,
                    AVSEEK_FLAG_BACKWARD) < 0) {
    return false;
  }
  avcodec_flush_buffers(dec->codec_ctx);
  dec->draining = false;

  // Decode forward until we reach or pass target_seconds
  while (VideoDecoder_next_frame(dec)) {
    if (dec->current_pts >= target_seconds) {
      return true;
    }
  }

  return false;
}

void VideoDecoder_close(VideoDecoder *dec) {
  cleanup_decoder(dec);
}
