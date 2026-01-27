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

// RTX Mega Geometry - Compute Cluster Tiling Shader Bindings
// Vulkan requires unique binding indices across all resource types in the same set

// Input SRVs (set 0)
#define COMPUTE_CLUSTER_TILING_VERTEX_CONTROL_POINTS     0
#define COMPUTE_CLUSTER_TILING_GEOMETRY_DATA             1
#define COMPUTE_CLUSTER_TILING_MATERIAL_CONSTANTS        2
#define COMPUTE_CLUSTER_TILING_SURFACE_TO_GEOMETRY       3
#define COMPUTE_CLUSTER_TILING_VERTEX_SURFACE_DESCS      4
#define COMPUTE_CLUSTER_TILING_VERTEX_CONTROL_INDICES    5
#define COMPUTE_CLUSTER_TILING_VERTEX_PATCH_OFFSETS      6
#define COMPUTE_CLUSTER_TILING_PLANS                     7
#define COMPUTE_CLUSTER_TILING_SUBPATCH_TREES            8
#define COMPUTE_CLUSTER_TILING_PATCH_POINT_INDICES       9
#define COMPUTE_CLUSTER_TILING_STENCIL_MATRIX           10
#define COMPUTE_CLUSTER_TILING_CLAS_INSTANTIATION_BYTES 11
#define COMPUTE_CLUSTER_TILING_TEMPLATE_ADDRESSES       12
#define COMPUTE_CLUSTER_TILING_TEXCOORD_SURFACE_DESCS   13
#define COMPUTE_CLUSTER_TILING_TEXCOORD_CONTROL_INDICES 14
#define COMPUTE_CLUSTER_TILING_TEXCOORD_PATCH_OFFSETS   15
#define COMPUTE_CLUSTER_TILING_TEXCOORDS                16

// Output UAVs (set 0) - Start at 17 to avoid conflicts with SRVs
#define COMPUTE_CLUSTER_TILING_GRID_SAMPLERS            17
#define COMPUTE_CLUSTER_TILING_TESSELLATION_COUNTERS    18
#define COMPUTE_CLUSTER_TILING_CLUSTERS                 19
#define COMPUTE_CLUSTER_TILING_CLUSTER_SHADING_DATA     20
#define COMPUTE_CLUSTER_TILING_INDIRECT_ARG_DATA        21
#define COMPUTE_CLUSTER_TILING_CLAS_ADDRESSES           22
#define COMPUTE_CLUSTER_TILING_VERTEX_PATCH_POINTS      23
#define COMPUTE_CLUSTER_TILING_TEXCOORD_PATCH_POINTS    24
#define COMPUTE_CLUSTER_TILING_DEBUG                    25

// Samplers (set 0) - Start at 26 to avoid conflicts with UAVs
#define COMPUTE_CLUSTER_TILING_DISPLACEMENT_SAMPLER     26
#define COMPUTE_CLUSTER_TILING_HIZ_SAMPLER              27

// Constant buffer
#define COMPUTE_CLUSTER_TILING_PARAMS                   28

// HIZ buffer array (set 1)
#define COMPUTE_CLUSTER_TILING_HIZ_BUFFER_ARRAY          0

// Bindless textures (separate set)
#define COMPUTE_CLUSTER_TILING_BINDLESS_TEXTURES_SET     2
#define COMPUTE_CLUSTER_TILING_BINDLESS_TEXTURES         0
