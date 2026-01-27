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

//!variant fill_instantiate_template_args.comp
//!end-variants

#include "rtx/pass/rtx_megageo/nvrhi_types.slangh"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_instantiate_template_args_params.h"
#include "rtx/pass/rtx_megageo/cluster_builder/fill_instantiate_template_args_binding_indices.h"

layout(push_constant)
ConstantBuffer<FillInstantiateTemplateArgsParams> cb;

layout(binding = FILL_INSTANTIATE_TEMPLATE_ARGS_TEMPLATE_ADDRESSES_INPUT)
StructuredBuffer<GpuVirtualAddress> InTemplateAddresses;

layout(binding = FILL_INSTANTIATE_TEMPLATE_ARGS_ARGS_OUTPUT)
RWStructuredBuffer<nvrhi::rt::cluster::IndirectInstantiateTemplateArgs> InOutInstantiateTemplateArgs;

[shader("compute")]
[numthreads(kFillInstantiateTemplateArgsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint templateIndex = threadIdx.x;
    if (templateIndex >= cb.numTemplates)
        return;

    nvrhi::rt::cluster::IndirectInstantiateTemplateArgs args = (nvrhi::rt::cluster::IndirectInstantiateTemplateArgs)0;
    args.clusterTemplate = InTemplateAddresses[templateIndex];
    args.vertexBuffer.startAddress = 0; // not providing vertex positions returns the worst case size
    args.vertexBuffer.strideInBytes = 0;
    InOutInstantiateTemplateArgs[templateIndex] = args;
}
