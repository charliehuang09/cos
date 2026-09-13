/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Hardware FD decode sequence adapted from NVIDIA's NvJpegDecoder.cpp.
// Keep setjmp/longjmp entirely in C: recovery must not skip C++ destructors.
#include "safe_nvjpeg_decoder.h"

#include <linux/videodev2.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libjpeg-9f/jpeglib.h"
#include "nvbufsurface.h"
#include "v4l2_nv_extensions.h"

struct RecoverableJpegError {
  struct jpeg_error_mgr manager;
  jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};

struct CosNvjpegDecoder {
  struct jpeg_decompress_struct cinfo;
  struct RecoverableJpegError error;
  int created;
};

static void jpeg_error_exit(j_common_ptr cinfo) {
  struct RecoverableJpegError* error = (struct RecoverableJpegError*)cinfo->err;
  (*cinfo->err->format_message)(cinfo, error->message);
  longjmp(error->jump, 1);
}

// A fresh decoder avoids retaining partially parsed tables or hardware state
// after an error. Destruction also has its own live recovery boundary.
static void reset_decoder(struct CosNvjpegDecoder* decoder) {
  if (decoder->created) {
    decoder->created = 0;
    if (setjmp(decoder->error.jump) == 0) {
      jpeg_destroy_decompress(&decoder->cinfo);
    }
  }
  memset(&decoder->cinfo, 0, sizeof(decoder->cinfo));
}

struct CosNvjpegDecoder* cos_nvjpeg_create(void) {
  return calloc(1, sizeof(struct CosNvjpegDecoder));
}

void cos_nvjpeg_destroy(struct CosNvjpegDecoder* decoder) {
  if (decoder != NULL) {
    reset_decoder(decoder);
    free(decoder);
  }
}

const char* cos_nvjpeg_error(const struct CosNvjpegDecoder* decoder) {
  return decoder->error.message;
}

int cos_nvjpeg_decode(struct CosNvjpegDecoder* decoder, unsigned char* data,
                      size_t size, int* fd, uint32_t* format,
                      uint32_t* width, uint32_t* height) {
  NvBufSurface surface = {0};
  *fd = -1;
  *format = *width = *height = 0;
  decoder->error.message[0] = '\0';
  if (setjmp(decoder->error.jump) != 0) {
    reset_decoder(decoder);
    return -1;
  }

  if (!decoder->created) {
    decoder->cinfo.err = jpeg_std_error(&decoder->error.manager);
    decoder->error.manager.error_exit = jpeg_error_exit;
    // jpeg_destroy_decompress also accepts a partially created object.
    decoder->created = 1;
    jpeg_create_decompress(&decoder->cinfo);
    decoder->cinfo.mjpeg_decode = TRUE;
    decoder->cinfo.dec_out_mem_type = NVBUF_MEM_SURFACE_ARRAY;
  }

  decoder->cinfo.out_color_space = JCS_YCbCr;
  jpeg_mem_src(&decoder->cinfo, data, size);
  if (jpeg_read_header(&decoder->cinfo, TRUE) != JPEG_HEADER_OK) {
    snprintf(decoder->error.message, sizeof(decoder->error.message),
             "Incomplete JPEG header");
    reset_decoder(decoder);
    return -1;
  }
  if (decoder->cinfo.num_components != 3) {
    snprintf(decoder->error.message, sizeof(decoder->error.message),
             "Unsupported JPEG component count: %d", decoder->cinfo.num_components);
    reset_decoder(decoder);
    return -1;
  }

  uint32_t pixel_format;
  if (decoder->cinfo.comp_info[0].h_samp_factor == 2) {
    pixel_format = decoder->cinfo.comp_info[0].v_samp_factor == 2
                       ? V4L2_PIX_FMT_YUV420M : V4L2_PIX_FMT_YUV422M;
  } else {
    pixel_format = decoder->cinfo.comp_info[0].v_samp_factor == 1
                       ? V4L2_PIX_FMT_YUV444M : V4L2_PIX_FMT_YUV422RM;
  }

  decoder->cinfo.out_color_space = JCS_YCbCr;
  decoder->cinfo.IsVendorbuf = TRUE;
  decoder->cinfo.pVendor_buf = (unsigned char*)&surface;
  jpeg_start_decompress(&decoder->cinfo);
  // NVIDIA's FD decoder returns to DSTATE_READY (202) after starting.
  if (decoder->cinfo.global_state != 202) {
    snprintf(decoder->error.message, sizeof(decoder->error.message),
             "JPEG format is not supported by the hardware decoder");
    reset_decoder(decoder);
    return -1;
  }
  jpeg_read_raw_data(&decoder->cinfo, NULL,
                    decoder->cinfo.comp_info[0].v_samp_factor * DCTSIZE);
  jpeg_finish_decompress(&decoder->cinfo);

  *width = (decoder->cinfo.image_width + 1U) & ~1U;
  *height = (decoder->cinfo.image_height + 1U) & ~1U;
  *format = pixel_format;
  *fd = decoder->cinfo.fd;
  return 0;
}
