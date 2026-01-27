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

// Component-wise min/max for float3 in C++
inline float3 min_float3(const float3& a, const float3& b) {
    return float3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

inline float3 max_float3(const float3& a, const float3& b) {
    return float3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
#endif

struct Box3
{
    // Alignment for constant buffer
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
#ifdef __cplusplus
        m_min = min_float3(v0, min_float3(v1, v2));
        m_max = max_float3(v0, max_float3(v1, v2));
#else
        m_min = min(v0, min(v1, v2));
        m_max = max(v0, max(v1, v2));
#endif
    }

    void Include(float3 p)
    {
#ifdef __cplusplus
        m_min = min_float3(m_min, p);
        m_max = max_float3(m_max, p);
#else
        m_min = min(m_min, p);
        m_max = max(m_max, p);
#endif
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

#ifdef __cplusplus
    Box3()
    {
        Init();
    }
#endif
};

#if defined(__cplusplus)
static_assert(sizeof(Box3) % 16 == 0);
#elif defined(TARGET_D3D12)
_Static_assert(sizeof(Box3) % 16 == 0, "Must be 16 byte aligned for constant buffer");
#endif

#endif // RTXMG_BOX3_H