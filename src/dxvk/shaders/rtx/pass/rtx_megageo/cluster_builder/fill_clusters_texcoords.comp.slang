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

// Separate shader for FillClustersTexcoordsMain entry point
//!variant fill_clusters_texcoords.comp
//!end-variants

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_clusters_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/copy_cluster_offset_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/tessellator_constants.h"
#include "rtx/pass/rtx_megageo/cluster_builder/cluster.h"
#include "rtxmg/subdivision/vertex.h"
#include "rtxmg/subdivision/subdivision_eval.hlsli"

// Vulkan bindings matching fill_clusters shared binding set (set 0)
[[vk::binding(0, 0)]] StructuredBuffer<GridSampler> t_GridSamplers : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<uint2> t_ClusterOffsetCounts : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<Cluster> t_Clusters : register(t2);

// Texcoord evaluation (bindings 15-19 in the shared layout)
[[vk::binding(15, 0)]] StructuredBuffer<LinearSurfaceDescriptor> t_TexCoordSurfaceDescriptors : register(t15);
[[vk::binding(16, 0)]] StructuredBuffer<Index> t_TexCoordControlPointIndices : register(t16);
[[vk::binding(17, 0)]] StructuredBuffer<uint32_t> t_TexCoordPatchPointsOffsets : register(t17);
[[vk::binding(18, 0)]] StructuredBuffer<float2> t_TexCoordPatchPoints : register(t18);
[[vk::binding(19, 0)]] StructuredBuffer<float2> t_TexCoords : register(t19);

// Output buffer (binding 22 in the shared layout - u_ClusterShadingData)
[[vk::binding(22, 0)]] RWStructuredBuffer<ClusterShadingData> u_ClusterShadingData : register(u1);

// Constant buffer (binding 25)
[[vk::binding(25, 0)]] ConstantBuffer<FillClustersParams> g_TessParams : register(b0);

void GathererWriteTexcoord(TexCoordLimitFrame texcoord, uint32_t clusterIndex, uint32_t cornerIndex)
{
    u_ClusterShadingData[clusterIndex].m_texcoords[cornerIndex] = texcoord.uv;
}

[shader("compute")]
[numthreads(kFillClustersTexcoordsThreadsX * 4, 1, 1)]
void main(uint3 threadIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID)
{
    const uint32_t clusterThreadIdx = threadIdx.x % kFillClustersTexcoordsThreadsX;
    const uint32_t cornerIdx = threadIdx.x / kFillClustersTexcoordsThreadsX;
    const uint32_t groupClusterIndex = groupIdx.x * kFillClustersTexcoordsThreadsX + clusterThreadIdx;
    uint2 offsetCount = t_ClusterOffsetCounts[g_TessParams.instanceIndex * (uint32_t)ClusterDispatchType::NumTypes + (uint32_t)ClusterDispatchType::All];
    if (groupClusterIndex >= offsetCount.y)
        return; // early out waves beyond cluster array end

    uint32_t clusterIndex = groupClusterIndex + offsetCount.x;
    const Cluster rCluster = t_Clusters[clusterIndex];
    const uint32_t iSurface = rCluster.iSurface;
    const GridSampler rSampler = t_GridSamplers[iSurface];

    // Extra shading data: texcoords on corners of surfaces (patches)
    const float2 kSurfaceUVs[4] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

    TexcoordEvaluatorHLSL texcoordEval;
    texcoordEval.m_surfaceDescriptors = t_TexCoordSurfaceDescriptors;
    texcoordEval.m_texcoordControlPointIndices = t_TexCoordControlPointIndices;
    texcoordEval.m_texcoordPatchPointsOffsets = t_TexCoordPatchPointsOffsets;
    texcoordEval.m_texcoordPatchPoints = t_TexCoordPatchPoints;
    texcoordEval.m_texcoordControlPoints = t_TexCoords;

    TexCoordLimitFrame texcoord = texcoordEval.EvaluateLinearSubd(kSurfaceUVs[cornerIdx], iSurface);

    GathererWriteTexcoord(texcoord, clusterIndex, cornerIdx);
}
