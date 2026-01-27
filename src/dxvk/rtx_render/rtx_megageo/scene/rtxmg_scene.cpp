/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "rtxmg_scene.h"
#include "../../util/log/log.h"

using namespace dxvk;

RTXMGScene::RTXMGScene(nvrhi::DeviceHandle device)
    : m_device(device)
{
    Initialize();
}

void RTXMGScene::Initialize() {
    CreateBuffers();
}

void RTXMGScene::CreateBuffers() {
    // Create geometry buffer (placeholder - actual size TBD)
    nvrhi::BufferDesc geometryDesc;
    geometryDesc.byteSize = 1024 * 1024; // 1MB default
    geometryDesc.debugName = "RTXMGScene_GeometryBuffer";
    geometryDesc.structStride = sizeof(float) * 16; // Example stride
    geometryDesc.canHaveUAVs = false;
    geometryDesc.canHaveRawViews = true;
    geometryDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    geometryDesc.keepInitialState = true;

    m_geometryBuffer = m_device->createBuffer(geometryDesc);

    // Create material buffer (placeholder)
    nvrhi::BufferDesc materialDesc;
    materialDesc.byteSize = 64 * 1024; // 64KB default
    materialDesc.debugName = "RTXMGScene_MaterialBuffer";
    materialDesc.structStride = sizeof(float) * 8; // Example stride
    materialDesc.canHaveUAVs = false;
    materialDesc.canHaveRawViews = true;
    materialDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    materialDesc.keepInitialState = true;

    m_materialBuffer = m_device->createBuffer(materialDesc);

    // Create displacement sampler
    nvrhi::SamplerDesc samplerDesc;
    samplerDesc.setAllFilters(true); // Linear filtering
    samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);

    m_displacementSampler = m_device->createSampler(samplerDesc);

    if (!m_geometryBuffer || !m_materialBuffer || !m_displacementSampler) {
        Logger::err("Failed to create RTXMGScene buffers");
    }
}

uint32_t RTXMGScene::TotalSubdPatchCount() const {
    // Sum SurfaceCount() for each instance (not just unique meshes)
    // This matches the sample's implementation and correctly accounts for
    // multiple instances of the same mesh
    const auto& instances = GetSubdMeshInstances();
    const auto& subds = GetSubdMeshes();

    uint32_t sum = 0;
    for (const auto& instance : instances) {
        if (instance.meshID < subds.size() && subds[instance.meshID]) {
            sum += subds[instance.meshID]->SurfaceCount();
        }
    }
    return sum;
}
