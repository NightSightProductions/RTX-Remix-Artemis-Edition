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

//!variant zbuffer_display.comp
//!end-variants

#include "rtx/pass/rtx_megageo/hiz/zbuffer_display_binding_indices.h"

layout(binding = ZBUFFER_DISPLAY_ZBUFFER_INPUT)
Texture2D<float> InZBuffer;

layout(binding = ZBUFFER_DISPLAY_MINMAX_INPUT)
StructuredBuffer<float> InMinMax;

layout(binding = ZBUFFER_DISPLAY_OUTPUT)
RWTexture2D<float4> InOutOutput;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadIdx : SV_GroupThreadID, uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint32_t width, height;
    InZBuffer.GetDimensions(width, height);

    uint32_t x = dispatchThreadId.x;
    uint32_t y = dispatchThreadId.y;

    if ((x >= width) || (y >= height))
    {
        return;
    }

    float depth = InZBuffer[dispatchThreadId];
    if (!isinf(depth))
    {
        float minz = InMinMax[0];
        float maxz = InMinMax[1];
        float deltaz = maxz - minz;
        depth = deltaz > 1e-6 ? (depth - minz) / deltaz : .5f;
        InOutOutput[dispatchThreadId] = float4(depth, depth, depth, 1);
    }
}
