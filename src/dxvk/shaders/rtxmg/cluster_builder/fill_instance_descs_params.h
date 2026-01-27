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
// OF THIS SOFTWARE, EVEN IFclust ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

static const uint32_t kFillInstanceDescsThreads = 32;

// Constant Buffer params
struct FillInstanceDescsParams
{
    uint32_t numInstances;
    uint32_t pad;
};

// RTX Remix: Params for patching only cluster BLAS addresses into instance buffer
// Used because RTX Remix has a mix of regular BLASes and cluster BLASes
struct PatchClusterBlasAddressParams
{
    uint32_t numMappings;          // Number of cluster instances to patch
    uint32_t instanceBufferStride; // sizeof(VkAccelerationStructureInstanceKHR)
    uint32_t pad0;
    uint32_t pad1;
};

// Mapping entry: RTX Remix instance index -> RTXMG blasPtrsBuffer index
struct ClusterInstanceMapping
{
    uint32_t remixInstanceIndex;   // Index in the VkAccelerationStructureInstanceKHR buffer
    uint32_t rtxmgInstanceIndex;   // Index in blasPtrsBuffer
};
