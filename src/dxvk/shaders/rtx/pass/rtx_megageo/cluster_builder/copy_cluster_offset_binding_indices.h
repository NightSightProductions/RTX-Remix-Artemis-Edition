/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

// RTX Mega Geometry - Copy Cluster Offset Shader Bindings
// Use HLSL register numbers - DXVK maps t/s/u/b registers to separate slot ranges:
//   t0-t99 (SRVs)     -> slots 0-99
//   s0-s99 (Samplers) -> slots 100-199
//   u0-u99 (UAVs)     -> slots 200-299
//   b0-b99 (CBVs)     -> slots 300-399

#define COPY_CLUSTER_OFFSET_TESS_COUNTERS_INPUT          0   // SRV t0
#define COPY_CLUSTER_OFFSET_CLUSTER_OFFSET_COUNTS_OUTPUT 0   // UAV u0
#define COPY_CLUSTER_OFFSET_FILL_INDIRECT_ARGS_OUTPUT    1   // UAV u1
#define COPY_CLUSTER_OFFSET_PARAMS                       0   // Constant buffer b0
