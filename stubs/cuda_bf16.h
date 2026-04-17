// Stub cuda_bf16.h for CUDA 10.2
// CUDA 10.2 does not ship cuda_bf16.h (bfloat16 support was added in CUDA 11).
// llama.cpp unconditionally includes it, so we provide a minimal stub that
// satisfies the compiler without implementing any BF16 operations.
// These types are never actually used on the code paths executed with CUDA 10.2.

#pragma once
#ifndef CUDA_BF16_H
#define CUDA_BF16_H

struct __nv_bfloat16  { unsigned short __x; };
struct __nv_bfloat162 { __nv_bfloat16 x; __nv_bfloat16 y; };

typedef __nv_bfloat16  nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

#endif // CUDA_BF16_H
