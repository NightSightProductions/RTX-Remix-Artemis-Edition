/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx/pass/common_binding_indices.h"

// =====================================================
// Photon Tracing Pass Bindings
// =====================================================

#define RESTIR_FG_TRACE_BINDING_PHOTON_BUFFER_OUTPUT          40
#define RESTIR_FG_TRACE_BINDING_PHOTON_AABB_GLOBAL_OUTPUT     41
#define RESTIR_FG_TRACE_BINDING_PHOTON_AABB_CAUSTIC_OUTPUT    42
#define RESTIR_FG_TRACE_BINDING_PHOTON_COUNTER                43

// =====================================================
// Photon Collection Pass Bindings
// =====================================================

// GBuffer inputs (matching ReSTIR GI pattern)
#define RESTIR_FG_COLLECT_BINDING_SHARED_FLAGS_INPUT                          40
#define RESTIR_FG_COLLECT_BINDING_SHARED_SURFACE_INDEX_INPUT                  41
#define RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT          42
#define RESTIR_FG_COLLECT_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT          43
#define RESTIR_FG_COLLECT_BINDING_PRIMARY_VIEW_DIRECTION_INPUT                44
#define RESTIR_FG_COLLECT_BINDING_PRIMARY_CONE_RADIUS_INPUT                   45
#define RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_POSITION_INPUT                46
#define RESTIR_FG_COLLECT_BINDING_PRIMARY_POSITION_ERROR_INPUT                47

// Photon data
#define RESTIR_FG_COLLECT_BINDING_PHOTON_AS                                   48
#define RESTIR_FG_COLLECT_BINDING_PHOTON_DATA                                 49
#define RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_GLOBAL                          50
#define RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_CAUSTIC                         51
#define RESTIR_FG_COLLECT_BINDING_PHOTON_COUNTER                              52

// Outputs
#define RESTIR_FG_COLLECT_BINDING_FG_RESERVOIR_OUTPUT                         53
#define RESTIR_FG_COLLECT_BINDING_FG_SAMPLE_OUTPUT                            54
#define RESTIR_FG_COLLECT_BINDING_CAUSTIC_RESERVOIR_OUTPUT                    55
#define RESTIR_FG_COLLECT_BINDING_CAUSTIC_SAMPLE_OUTPUT                       56
#define RESTIR_FG_COLLECT_BINDING_SURFACE_DATA_OUTPUT                         57

// =====================================================
// Final Gather Resampling Pass Bindings
// =====================================================

#define RESTIR_FG_RESAMPLE_BINDING_MVEC_INPUT                 40
#define RESTIR_FG_RESAMPLE_BINDING_WORLD_POSITION_INPUT       41
#define RESTIR_FG_RESAMPLE_BINDING_WORLD_NORMAL_INPUT         42
#define RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_PREV             43
#define RESTIR_FG_RESAMPLE_BINDING_SAMPLE_PREV                44
#define RESTIR_FG_RESAMPLE_BINDING_SURFACE_PREV               45
#define RESTIR_FG_RESAMPLE_BINDING_SURFACE_CURR               46
#define RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_CURR             47
#define RESTIR_FG_RESAMPLE_BINDING_SAMPLE_CURR                48

// =====================================================
// Caustic Resampling Pass Bindings
// =====================================================

#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_MVEC_INPUT           40
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_POSITION_INPUT 41
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_NORMAL_INPUT   42
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_PREV       43
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_PREV          44
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SURFACE_PREV         45
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SURFACE_CURR         46
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_CURR       47
#define RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_CURR          48

// =====================================================
// Final Shading Pass Bindings
// =====================================================

#define RESTIR_FG_FINAL_SHADING_BINDING_WORLD_POSITION_INPUT        40
#define RESTIR_FG_FINAL_SHADING_BINDING_WORLD_NORMAL_INPUT          41
#define RESTIR_FG_FINAL_SHADING_BINDING_PERCEPTUAL_ROUGHNESS_INPUT  42
#define RESTIR_FG_FINAL_SHADING_BINDING_ALBEDO_INPUT                43
#define RESTIR_FG_FINAL_SHADING_BINDING_FG_RESERVOIR                44
#define RESTIR_FG_FINAL_SHADING_BINDING_FG_SAMPLE                   45
#define RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_RESERVOIR           46
#define RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_SAMPLE              47
#define RESTIR_FG_FINAL_SHADING_BINDING_OUTPUT                      48
#define RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_OUTPUT              49
