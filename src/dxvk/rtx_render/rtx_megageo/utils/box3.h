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

#ifndef RTXMG_BOX3_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RTXMG_BOX3_H

#ifdef __cplusplus
#include <algorithm>  // For std::min, std::max

// Use float4 for constant buffer compatibility (C++ float3 is 16 bytes, breaking alignment)
// This ensures Box3 is exactly 32 bytes in both C++ and HLSL
struct Box3
{
    float4 m_min;  // xyz = min, w = padding
    float4 m_max;  // xyz = max, w = padding

    void Init()
    {
        m_min = float4(1e37f, 1e37f, 1e37f, 0.0f);
        m_max = float4(-1e37f, -1e37f, -1e37f, 0.0f);
    }

    void Init(float3 v0, float3 v1, float3 v2)
    {
        float3 minV(std::min({v0.x, v1.x, v2.x}), std::min({v0.y, v1.y, v2.y}), std::min({v0.z, v1.z, v2.z}));
        float3 maxV(std::max({v0.x, v1.x, v2.x}), std::max({v0.y, v1.y, v2.y}), std::max({v0.z, v1.z, v2.z}));
        m_min = float4(minV.x, minV.y, minV.z, 0.0f);
        m_max = float4(maxV.x, maxV.y, maxV.z, 0.0f);
    }

    void Include(float3 p)
    {
        m_min = float4(std::min(m_min.x, p.x), std::min(m_min.y, p.y), std::min(m_min.z, p.z), 0.0f);
        m_max = float4(std::max(m_max.x, p.x), std::max(m_max.y, p.y), std::max(m_max.z, p.z), 0.0f);
    }

    float3 Extent()
    {
        return float3(m_max.x - m_min.x, m_max.y - m_min.y, m_max.z - m_min.z);
    }

    bool Valid()
    {
        return m_min.x <= m_max.x &&
            m_min.y <= m_max.y &&
            m_min.z <= m_max.z;
    }

    Box3()
    {
        Init();
    }
};

static_assert(sizeof(Box3) == 32, "Box3 must be exactly 32 bytes to match HLSL layout");

#else
// HLSL/Slang version - uses float3 + pad which is naturally 32 bytes
struct Box3
{
    float3 m_min;
    float pad0;
    float3 m_max;
    float pad1;

    void Init()
    {
        m_min = float3(1e37f, 1e37f, 1e37f);
        m_max = float3(-1e37f, -1e37f, -1e37f);
        pad0 = 0; pad1 = 0;
    }

    void Init(float3 v0, float3 v1, float3 v2)
    {
        m_min = min(v0, min(v1, v2));
        m_max = max(v0, max(v1, v2));
    }

    void Include(float3 p)
    {
        m_min = min(m_min, p);
        m_max = max(m_max, p);
    }

    float3 Extent()
    {
        return m_max - m_min;
    }

    bool Valid()
    {
        return m_min.x <= m_max.x &&
            m_min.y <= m_max.y &&
            m_min.z <= m_max.z;
    }
};
#endif

#if defined(TARGET_D3D12)
_Static_assert(sizeof(Box3) % 16 == 0, "Must be 16 byte aligned for constant buffer");
#endif

#endif // RTXMG_BOX3_H