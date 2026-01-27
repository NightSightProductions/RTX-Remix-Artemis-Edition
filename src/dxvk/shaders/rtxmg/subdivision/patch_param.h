//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// NOTE: Converted from C++ member function syntax to Slang-compatible free functions

#pragma once

#include "rtxmg/subdivision/osd_ports/tmr/types.h"

struct PatchParam
{
    uint field0;
    uint field1;
};

// Helper functions (previously private members)
uint PatchParam_pack(uint value, int width, int offset)
{
    return (uint)((value & ((1U << width) - 1)) << offset);
}

uint PatchParam_unpack(uint value, int width, int offset)
{
    return (uint)((value >> offset) & ((1U << width) - 1));
}

// Public interface (previously member functions)

/// \brief Resets everything to 0
void PatchParam_Clear(inout PatchParam param)
{
    param.field0 = 0;
    param.field1 = 0;
}

/// \brief Returns the faceid
Index PatchParam_GetFaceId(PatchParam param)
{
    return Index(PatchParam_unpack(param.field0, 28, 0));
}

/// \brief Returns the log2 value of the u parameter at the first corner of the patch
uint16_t PatchParam_GetU(PatchParam param)
{
    return (uint16_t)PatchParam_unpack(param.field1, 10, 22);
}

/// \brief Returns the log2 value of the v parameter at the first corner of the patch
uint16_t PatchParam_GetV(PatchParam param)
{
    return (uint16_t)PatchParam_unpack(param.field1, 10, 12);
}

/// \brief Returns the transition edge encoding for the patch.
uint16_t PatchParam_GetTransition(PatchParam param)
{
    return (uint16_t)PatchParam_unpack(param.field0, 4, 28);
}

/// \brief Returns the boundary edge encoding for the patch.
uint16_t PatchParam_GetBoundary(PatchParam param)
{
    return (uint16_t)PatchParam_unpack(param.field1, 5, 7);
}

/// \brief True if the parent base face is a non-quad
bool PatchParam_NonQuadRoot(PatchParam param)
{
    return (PatchParam_unpack(param.field1, 1, 4) != 0);
}

/// \brief Returns the level of subdivision of the patch
uint16_t PatchParam_GetDepth(PatchParam param)
{
    return (uint16_t)PatchParam_unpack(param.field1, 4, 0);
}

/// \brief Returns whether the patch is regular
bool PatchParam_IsRegular(PatchParam param)
{
    return (PatchParam_unpack(param.field1, 1, 5) != 0);
}

/// \brief Returns the fraction of unit parametric space covered by this face.
float PatchParam_GetParamFraction(PatchParam param)
{
    return 1.0f / (float)(1U << (PatchParam_GetDepth(param) - PatchParam_NonQuadRoot(param)));
}

/// \brief Returns if a triangular patch is parametrically rotated 180 degrees
bool PatchParam_IsTriangleRotated(PatchParam param)
{
    return (PatchParam_GetU(param) + PatchParam_GetV(param)) >= (1U << PatchParam_GetDepth(param));
}

/// \brief Sets the values of the bit fields
void PatchParam_Set(inout PatchParam param, Index faceid, uint16_t u, uint16_t v,
    uint16_t depth, bool nonquad,
    uint16_t boundary, uint16_t transition,
    bool regular)
{
    param.field0 = PatchParam_pack(faceid, 28, 0) |
        PatchParam_pack(transition, 4, 28);

    param.field1 = PatchParam_pack(u, 10, 22) |
        PatchParam_pack(v, 10, 12) |
        PatchParam_pack(boundary, 5, 7) |
        PatchParam_pack(regular, 1, 5) |
        PatchParam_pack(nonquad, 1, 4) |
        PatchParam_pack(depth, 4, 0);
}

/// \brief A (u,v) pair in the fraction of parametric space covered by this
/// face is mapped into a normalized parametric space.
void PatchParam_Normalize(PatchParam param, inout float u, inout float v)
{
    float fracInv = (float)(1.0f / PatchParam_GetParamFraction(param));

    u = u * fracInv - (float)PatchParam_GetU(param);
    v = v * fracInv - (float)PatchParam_GetV(param);
}

/// \brief A (u,v) pair in a normalized parametric space is mapped back into the
/// fraction of parametric space covered by this face.
void PatchParam_Unnormalize(PatchParam param, inout float u, inout float v)
{
    float frac = (float)PatchParam_GetParamFraction(param);

    u = (u + (float)PatchParam_GetU(param)) * frac;
    v = (v + (float)PatchParam_GetV(param)) * frac;
}

/// \brief Normalize for triangular patches
void PatchParam_NormalizeTriangle(PatchParam param, inout float u, inout float v)
{
    if (PatchParam_IsTriangleRotated(param))
    {
        float fracInv = (float)(1.0f / PatchParam_GetParamFraction(param));

        int depthFactor = 1U << PatchParam_GetDepth(param);
        u = (float)(depthFactor - PatchParam_GetU(param)) - (u * fracInv);
        v = (float)(depthFactor - PatchParam_GetV(param)) - (v * fracInv);
    }
    else
    {
        PatchParam_Normalize(param, u, v);
    }
}

/// \brief Unnormalize for triangular patches
void PatchParam_UnnormalizeTriangle(PatchParam param, inout float u, inout float v)
{
    if (PatchParam_IsTriangleRotated(param))
    {
        float frac = PatchParam_GetParamFraction(param);

        int depthFactor = 1U << PatchParam_GetDepth(param);
        u = ((float)(depthFactor - PatchParam_GetU(param)) - u) * frac;
        v = ((float)(depthFactor - PatchParam_GetV(param)) - v) * frac;
    }
    else
    {
        PatchParam_Unnormalize(param, u, v);
    }
}
