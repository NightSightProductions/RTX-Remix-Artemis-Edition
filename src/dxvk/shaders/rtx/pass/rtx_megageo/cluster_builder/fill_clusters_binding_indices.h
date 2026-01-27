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

// RTX Mega Geometry - Fill Clusters Shader Bindings

// Push constant / Constant buffer
#define FILL_CLUSTERS_PARAMS                             0

// Input SRVs (set 0)
#define FILL_CLUSTERS_GRID_SAMPLERS                      0
#define FILL_CLUSTERS_CLUSTER_OFFSET_COUNTS              1
#define FILL_CLUSTERS_CLUSTERS                           2
#define FILL_CLUSTERS_VERTEX_CONTROL_POINTS              3
#define FILL_CLUSTERS_VERTEX_SURFACE_DESCS               4
#define FILL_CLUSTERS_VERTEX_CONTROL_INDICES             5
#define FILL_CLUSTERS_VERTEX_PATCH_OFFSETS               6
#define FILL_CLUSTERS_PLANS                              7
#define FILL_CLUSTERS_SUBPATCH_TREES                     8
#define FILL_CLUSTERS_PATCH_POINT_INDICES                9
#define FILL_CLUSTERS_STENCIL_MATRIX                    10
#define FILL_CLUSTERS_VERTEX_PATCH_POINTS               11
#define FILL_CLUSTERS_GEOMETRY_DATA                     12
#define FILL_CLUSTERS_MATERIAL_CONSTANTS                13
#define FILL_CLUSTERS_SURFACE_TO_GEOMETRY               14
#define FILL_CLUSTERS_TEXCOORD_SURFACE_DESCS            15
#define FILL_CLUSTERS_TEXCOORD_CONTROL_INDICES          16
#define FILL_CLUSTERS_TEXCOORD_PATCH_OFFSETS            17
#define FILL_CLUSTERS_TEXCOORD_PATCH_POINTS             18
#define FILL_CLUSTERS_TEXCOORDS                         19

// Output UAVs (set 0) - Start after SRVs to avoid binding conflicts
#define FILL_CLUSTERS_CLUSTER_VERTEX_POSITIONS          20
#define FILL_CLUSTERS_CLUSTER_SHADING_DATA              21
#define FILL_CLUSTERS_DEBUG                             22
#define FILL_CLUSTERS_CLUSTER_VERTEX_NORMALS            23

// Samplers (set 0) - Start after UAVs to avoid binding conflicts
#define FILL_CLUSTERS_DISPLACEMENT_SAMPLER              24

// Constant buffer
#define FILL_CLUSTERS_PARAMS_CB                         25

// Bindless textures (separate set)
#define FILL_CLUSTERS_BINDLESS_TEXTURES_SET              1
#define FILL_CLUSTERS_BINDLESS_TEXTURES                  0
