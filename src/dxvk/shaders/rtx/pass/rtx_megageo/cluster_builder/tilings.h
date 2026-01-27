//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// NOTE: Converted from C++ member function syntax to Slang-compatible free functions
//
#pragma once

struct ClusterTiling
{
    uint16_t2 tilingSize;    // number of tiles in x and y direction
    uint16_t2 clusterSize;   // number of quads in x and y direction inside tile
};

// ClusterTiling free functions (previously member functions)

uint32_t ClusterTiling_ClusterCount(ClusterTiling tiling)
{
    return uint32_t(tiling.tilingSize.x) * uint32_t(tiling.tilingSize.y);
}

uint32_t ClusterTiling_ClusterVertexCount(ClusterTiling tiling)
{
    return (tiling.clusterSize.x + 1) * (tiling.clusterSize.y + 1);
}

uint32_t ClusterTiling_VertexCount(ClusterTiling tiling)
{
    return ClusterTiling_ClusterVertexCount(tiling) * ClusterTiling_ClusterCount(tiling);
}

uint16_t2 ClusterTiling_ClusterIndex2D(ClusterTiling tiling, uint32_t rowMajorIndex)
{
    return uint16_t2((uint16_t)(rowMajorIndex % tiling.tilingSize.x),
                     (uint16_t)(rowMajorIndex / tiling.tilingSize.x));
}

uint16_t2 ClusterTiling_QuadOffset2D(ClusterTiling tiling, uint32_t rowMajorIndex)
{
    return ClusterTiling_ClusterIndex2D(tiling, rowMajorIndex) * uint16_t2(tiling.clusterSize.x, tiling.clusterSize.y);
}

uint2 ClusterTiling_VertexIndex2D(ClusterTiling tiling, uint32_t rowMajorIndex)
{
    uint32_t verticesU = tiling.clusterSize.x + 1;
    return uint2(rowMajorIndex % verticesU, rowMajorIndex / verticesU);
}

// SurfaceTiling constants (previously enum)
static const int SurfaceTiling_REGULAR = 0;
static const int SurfaceTiling_RIGHT = 1;
static const int SurfaceTiling_TOP = 2;
static const int SurfaceTiling_CORNER = 3;
static const int SurfaceTiling_N_SUB_TILINGS = 4;

struct SurfaceTiling
{
    ClusterTiling subTilings[4];    // [N_SUB_TILINGS]
    uint16_t2 quadOffsets[4];       // quad offset of the tiling in x and y direction
};

// SurfaceTiling free functions (previously member functions)

uint32_t SurfaceTiling_ClusterCount(SurfaceTiling tiling)
{
    uint32_t sum = 0;
    for (int iTiling = 0; iTiling < SurfaceTiling_N_SUB_TILINGS; ++iTiling)
        sum += ClusterTiling_ClusterCount(tiling.subTilings[iTiling]);
    return sum;
}

uint32_t SurfaceTiling_VertexCount(SurfaceTiling tiling)
{
    uint32_t sum = 0;
    for (int iTiling = 0; iTiling < SurfaceTiling_N_SUB_TILINGS; ++iTiling)
        sum += ClusterTiling_VertexCount(tiling.subTilings[iTiling]);
    return sum;
}

uint16_t2 SurfaceTiling_ClusterOffset(SurfaceTiling tiling, uint16_t iTiling, uint32_t iCluster)
{
    return tiling.quadOffsets[iTiling] + ClusterTiling_QuadOffset2D(tiling.subTilings[iTiling], iCluster);
}

// Factory function for creating SurfaceTiling
SurfaceTiling MakeSurfaceTiling(uint16_t2 surfaceSize)
{
    SurfaceTiling ret;
    uint16_t targetEdgeSegments = 8;

    uint16_t2 regularGridSize;
    uint16_t2 modCluster;
    {
        uint16_t2 divClusters = uint16_t2((uint16_t)(surfaceSize.x / targetEdgeSegments),
            (uint16_t)(surfaceSize.y / targetEdgeSegments));
        modCluster = uint16_t2((uint16_t)(surfaceSize.x % targetEdgeSegments),
            (uint16_t)(surfaceSize.y % targetEdgeSegments));

        uint32_t maxEdgeSegments = kMaxClusterEdgeSegments;
        if (divClusters.x > 0 && modCluster.x + targetEdgeSegments <= maxEdgeSegments)
        {
            divClusters.x -= 1;
            modCluster.x += targetEdgeSegments;
        }
        if (divClusters.y > 0 && modCluster.y + targetEdgeSegments <= maxEdgeSegments)
        {
            divClusters.y -= 1;
            modCluster.y += targetEdgeSegments;
        }
        regularGridSize = divClusters;
    }

    ret.subTilings[SurfaceTiling_REGULAR].tilingSize = regularGridSize;
    ret.subTilings[SurfaceTiling_REGULAR].clusterSize = uint16_t2(targetEdgeSegments, targetEdgeSegments);
    ret.quadOffsets[SurfaceTiling_REGULAR] = uint16_t2(0u, 0u);

    ret.subTilings[SurfaceTiling_RIGHT].tilingSize = uint16_t2(1u, regularGridSize.y);
    ret.subTilings[SurfaceTiling_RIGHT].clusterSize = uint16_t2(modCluster.x, targetEdgeSegments);
    ret.quadOffsets[SurfaceTiling_RIGHT] = uint16_t2((uint16_t)(regularGridSize.x * targetEdgeSegments), 0u);

    ret.subTilings[SurfaceTiling_TOP].tilingSize = uint16_t2(regularGridSize.x, 1u);
    ret.subTilings[SurfaceTiling_TOP].clusterSize = uint16_t2(targetEdgeSegments, modCluster.y);
    ret.quadOffsets[SurfaceTiling_TOP] = uint16_t2(0u, (uint16_t)(regularGridSize.y * targetEdgeSegments));

    ret.subTilings[SurfaceTiling_CORNER].tilingSize = uint16_t2(1u, 1u);
    ret.subTilings[SurfaceTiling_CORNER].clusterSize = modCluster;
    ret.quadOffsets[SurfaceTiling_CORNER] = uint16_t2((uint16_t)(regularGridSize.x * targetEdgeSegments),
        (uint16_t)(regularGridSize.y * targetEdgeSegments));

    return ret;
}
