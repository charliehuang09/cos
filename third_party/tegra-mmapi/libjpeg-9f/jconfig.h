/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#define HAVE_PROTOTYPES 1
#define HAVE_UNSIGNED_CHAR 1
#define HAVE_UNSIGNED_SHORT 1
#define HAVE_STDDEF_H 1
#define HAVE_STDLIB_H 1
#define HAVE_LOCALE_H 1

#ifdef _WIN32
#ifndef __RPCNDR_H__
typedef unsigned char boolean;
#endif
#define HAVE_BOOLEAN
#endif

#ifdef JPEG_INTERNALS
#define INLINE __inline__
#endif

#ifndef TEGRA_ACCELERATE
#if !defined(GPU_ACCELERATE)
#define TEGRA_ACCELERATE
#endif
#endif

#ifdef JPEG_CJPEG_DJPEG
#define BMP_SUPPORTED
#define GIF_SUPPORTED
#define PPM_SUPPORTED
#define TARGA_SUPPORTED
#endif
