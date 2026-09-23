#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Serialized by the caller. Output FDs are owned by the decoder and remain
// valid only until its next decode/reset/destruction; callers must copy them.
struct CosNvjpegDecoder;
struct CosNvjpegDecoder* cos_nvjpeg_create(void);
void cos_nvjpeg_destroy(struct CosNvjpegDecoder* decoder);
// Returns 0 on success; -1 drops the frame and resets the decoder. Read the
// diagnostic with cos_nvjpeg_error before the next decode.
int cos_nvjpeg_decode(struct CosNvjpegDecoder* decoder, unsigned char* data,
                      size_t size, int* fd, uint32_t* format,
                      uint32_t* width, uint32_t* height);
const char* cos_nvjpeg_error(const struct CosNvjpegDecoder* decoder);

#ifdef __cplusplus
}
#endif
