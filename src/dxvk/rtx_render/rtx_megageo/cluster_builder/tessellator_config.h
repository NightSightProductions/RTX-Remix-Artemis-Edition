/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "cluster.h"
#include "../scene/camera.h"

class ZBuffer;

struct TessellatorConfig
{
    static constexpr float kDefaultFineTessellationRate = 1.0f;
    static constexpr float kDefaultCoarseTessellationRate = 1.0f / 15.0f;

    // REDUCED: 256K clusters for better memory compatibility (was 2M)
    static constexpr uint32_t kDefaultMaxClusters = (1u << 18);

    // REDUCED: 128MB per vertex buffer for better memory compatibility (was 512MB)
    static constexpr size_t kDefaultVertexBufferBytes = (128ull << 20);

    // REDUCED: 256MB CLAS memory for better memory compatibility (was 1GB)
    static constexpr size_t kDefaultClasBufferBytes = (256ull << 20);

    static constexpr uint32_t kMinIsolationLevel = 1u;
    static constexpr uint32_t kMaxIsolationLevel = 6u;

    enum class VisibilityMode
    {
        VIS_LIMIT_EDGES = 0,
        VIS_SURFACE = 1,
        COUNT
    };

    enum class AdaptiveTessellationMode
    {
        UNIFORM = 0,
        WORLD_SPACE_EDGE_LENGTH,
        SPHERICAL_PROJECTION,
        COUNT
    };
    
    struct MemorySettings
    {
        uint32_t maxClusters = kDefaultMaxClusters;
        size_t clasBufferBytes = kDefaultClasBufferBytes;
        size_t vertexBufferBytes = kDefaultVertexBufferBytes;

        bool operator==(const MemorySettings& o) const
        {
            return vertexBufferBytes == o.vertexBufferBytes &&
                maxClusters == o.maxClusters &&
                clasBufferBytes == o.clasBufferBytes;
        }
    };
    
    MemorySettings memorySettings;
    VisibilityMode visMode = VisibilityMode::VIS_LIMIT_EDGES;
    AdaptiveTessellationMode tessMode = AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH;

    float fineTessellationRate = kDefaultFineTessellationRate;
    float coarseTessellationRate = kDefaultCoarseTessellationRate;
    bool  enableFrustumVisibility = true;
    bool  enableHiZVisibility = true;
    bool  enableBackfaceVisibility = true;
    bool  enableLogging = false; // enable debug logging for tessellator build
    bool  enableMonolithicClusterBuild = false;
    bool  enableVertexNormals = false; // enable vertex normal computation

    uint2            viewportSize = { 0u, 0u };
    uint4            edgeSegments = { 8, 8, 8, 8 };
    uint32_t         isolationLevel = 0; // 0 is dynamic, >0 is fixed
    ClusterPattern   clusterPattern = ClusterPattern::SLANTED;
    unsigned char    quantNBits = 0;

    float            displacementScale = 1.0f;

    const dxvk::Camera* camera = nullptr;
    const ZBuffer* zbuffer = nullptr;

    int debugSurfaceIndex = -1;  // -1 to disable debug
    int debugClusterIndex = -1;
    int debugLaneIndex = -1;
};

#if __cplusplus
#include <array>
constexpr std::array<const char*, 3> kAdaptiveTessellationModeNames = {
    "Uniform",
    "WS Edge Length",
    "Spherical Projection"
};
static_assert(kAdaptiveTessellationModeNames.size() == size_t(TessellatorConfig::AdaptiveTessellationMode::COUNT));

constexpr std::array<const char*, 2> kVisibilityModeNames = {
    "Limit Edge",
    "Surface 1-Ring"
};
static_assert(kVisibilityModeNames.size() == size_t(TessellatorConfig::VisibilityMode::COUNT));
#endif