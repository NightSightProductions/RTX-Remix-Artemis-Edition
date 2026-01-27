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

//!variant fill_blas_from_clas_args.comp
//!end-variants

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_blas_from_clas_args_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/copy_cluster_offset_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_blas_from_clas_args_binding_indices.h"

layout(push_constant)
ConstantBuffer<FillBlasFromClasArgsParams> cb;

layout(binding = FILL_BLAS_FROM_CLAS_ARGS_CLUSTER_OFFSET_COUNTS_INPUT)
StructuredBuffer<uint2> InClusterOffsetCounts;

layout(binding = FILL_BLAS_FROM_CLAS_ARGS_BLAS_ARGS_OUTPUT)
RWStructuredBuffer<nvrhi::rt::cluster::IndirectArgs> InOutBlasFromClasArgs;

[shader("compute")]
[numthreads(kFillBlasFromClasArgsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t instanceIndex = threadIdx.x;
    if (instanceIndex >= cb.numInstances)
        return;

    uint2 offsetCount = InClusterOffsetCounts[instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::All];

    // Initialize to zero like the sample to avoid uninitialized memory
    nvrhi::rt::cluster::IndirectArgs args = (nvrhi::rt::cluster::IndirectArgs)0;
    args.clusterCount = offsetCount.y;
    // Note: clusterReferencesStride is left as 0 to match sample behavior
    // D3D12 NVRHI calls this 'reserved' (must be 0), Vulkan uses it as stride
    // 0 value means tightly packed (driver uses default sizeof(VkDeviceAddress))
    args.clusterAddresses = cb.clasAddressesBaseAddress + uint64_t(8) * offsetCount.x;  // sizeof(GpuVirtualAddress) = 8
    InOutBlasFromClasArgs[instanceIndex] = args;
}
