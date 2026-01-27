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

//!variant fill_instance_descs.comp
//!end-variants

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_instance_descs_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_instance_descs_binding_indices.h"

layout(push_constant)
ConstantBuffer<FillInstanceDescsParams> cb;

layout(binding = FILL_INSTANCE_DESCS_BLAS_ADDRESSES_INPUT)
StructuredBuffer<GpuVirtualAddress> InBlasAddresses;

layout(binding = FILL_INSTANCE_DESCS_INSTANCE_DESCS_OUTPUT)
RWStructuredBuffer<nvrhi::rt::IndirectInstanceDesc> InOutInstanceDescs;

[shader("compute")]
[numthreads(kFillInstanceDescsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t instanceIndex = threadIdx.x;
    if (instanceIndex >= cb.numInstances)
        return;

    GpuVirtualAddress blasAddress = InBlasAddresses[instanceIndex];

    nvrhi::rt::IndirectInstanceDesc instanceDesc = InOutInstanceDescs[instanceIndex];
    instanceDesc.blasDeviceAddress = blasAddress;
    InOutInstanceDescs[instanceIndex] = instanceDesc;
}
