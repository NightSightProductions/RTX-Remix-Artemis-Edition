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

//!variant hiz_display.comp
//!end-variants

#include "rtx/pass/rtx_megageo/hiz/hiz_buffer_constants.h"
#include "rtx/pass/rtx_megageo/hiz/hiz_buffer_display_params.h"
#include "rtx/pass/rtx_megageo/hiz/hiz_display_binding_indices.h"

layout(binding = HIZ_DISPLAY_PARAMS_INPUT)
ConstantBuffer<HiZDisplayParams> InParams;

layout(binding = HIZ_DISPLAY_HIZ_ARRAY)
Texture2D<float> InHiZ[HIZ_MAX_LODS];

layout(binding = HIZ_DISPLAY_OUTPUT)
RWTexture2D<float4> InOutOutput;

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint32_t width, height;
    InHiZ[InParams.level].GetDimensions(width, height);

    uint32_t outWidth, outHeight;
    InOutOutput.GetDimensions(outWidth, outHeight);

    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    float depth = InHiZ[InParams.level][dispatchThreadId];

    uint32_t2 outputIdx = uint32_t2(x + InParams.offsetX, outHeight + y - height - InParams.offsetY);

    if (isinf(depth))
    {
        InOutOutput[outputIdx] = float4(1, 0, 0, 0);
    }
    else
    {
        depth = (depth <= 0.f) ? 1.f : 1.f / depth;
        InOutOutput[outputIdx] = float4(depth, depth, depth, 1);
    }
}
