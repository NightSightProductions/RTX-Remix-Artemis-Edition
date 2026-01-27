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

#include "zbuffer.h"
#include "hiz_buffer.h"

#include "../utils/buffer.h"
#include "../utils/debug.h"
#include "../nvrhi_adapter/nvrhi_dxvk_shader.h"

#include "../../../util/log/log.h"
#include "../../rtx_shader_manager.h"

#include <rtx_shaders/zbuffer_minmax.h>
#include <rtx_shaders/zbuffer_display.h>
#include "../../shaders/rtx/pass/rtx_megageo/hiz/zbuffer_minmax_binding_indices.h"
#include "../../shaders/rtx/pass/rtx_megageo/hiz/zbuffer_display_binding_indices.h"

#include <fstream>

using namespace dxvk;

// Shader definitions for RTX MG ZBuffer
namespace {
    class ZBufferMinMaxShader : public ManagedShader
    {
        SHADER_SOURCE(ZBufferMinMaxShader, VK_SHADER_STAGE_COMPUTE_BIT, zbuffer_minmax)

        BEGIN_PARAMETER()
            TEXTURE2D(ZBUFFER_MINMAX_ZBUFFER_INPUT)
            RW_STRUCTURED_BUFFER(ZBUFFER_MINMAX_MINMAX_OUTPUT)
        END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ZBufferMinMaxShader);

    class ZBufferDisplayShader : public ManagedShader
    {
        SHADER_SOURCE(ZBufferDisplayShader, VK_SHADER_STAGE_COMPUTE_BIT, zbuffer_display)

        BEGIN_PARAMETER()
            TEXTURE2D(ZBUFFER_DISPLAY_ZBUFFER_INPUT)
            STRUCTURED_BUFFER(ZBUFFER_DISPLAY_MINMAX_INPUT)
            RW_TEXTURE2D(ZBUFFER_DISPLAY_OUTPUT)
        END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ZBufferDisplayShader);
}

std::unique_ptr<ZBuffer> ZBuffer::Create(uint2 size,
    nvrhi::IDevice* device,
    nvrhi::ICommandList* commandList)
{
    auto zbuffer = std::make_unique<ZBuffer>();

    nvrhi::TextureDesc desc;
    desc.width = size.x;
    desc.height = size.y;
    desc.isUAV = true;
    desc.keepInitialState = true;
    desc.format = nvrhi::Format::R32_FLOAT;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.debugName = "ZBuffer";
    zbuffer->m_currentTexture = device->createTexture(desc);
    zbuffer->m_minmaxBuffer.Create(2, "ZBufferMinMax", device);

    // Load shaders using RTX Remix ShaderManager
    zbuffer->m_minmaxShader = wrapShader(ZBufferMinMaxShader::getShader());
    zbuffer->m_displayShader = wrapShader(ZBufferDisplayShader::getShader());

    if (!zbuffer->m_minmaxShader || !zbuffer->m_displayShader)
    {
        dxvk::Logger::err("ZBuffer: Failed to load shaders");
        return zbuffer;
    }

    // make level 0 size a multiple of 32 to avoid nasty edge cases
    uint2  hizSize = { (((uint32_t(size.x) + 255) & ~255u) / 8),
                      (((uint32_t(size.y) + 255) & ~255u) / 8) };
    zbuffer->m_hierarchy = HiZBuffer::Create(hizSize, device, commandList);

    return zbuffer;
}

void ZBuffer::Display(nvrhi::ITexture* output, nvrhi::ICommandList* commandList)
{
    if (!m_currentTexture) return;

    nvrhi::utils::ScopedMarker marker(commandList, "ZBuffer::display");
    uint2 size = { m_currentTexture->getDesc().width, m_currentTexture->getDesc().height };

    uint32_t minmax[2] = { std::numeric_limits<uint32_t>::max(), 0 };
    commandList->writeBuffer(m_minmaxBuffer.GetBuffer(), minmax, sizeof(minmax));

    auto device = commandList->getDevice();

    // MinMax pass
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_currentTexture))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_minmaxBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_minmaxBL, bindingSet))
    {
        dxvk::Logger::err("ZBuffer: Failed to create binding set for minmax");
        return;
    }

    if (!m_minmaxPSO)
    {
        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_minmaxShader)
            .addBindingLayout(m_minmaxBL);

        m_minmaxPSO = device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_minmaxPSO)
        .addBindingSet(bindingSet);

    commandList->setComputeState(state);
    commandList->dispatch(size.x / 16 + 1, size.y / 16 + 1, 1);

    // Display pass
    bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_currentTexture))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_minmaxBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, output));

    bindingSet = nullptr;
    if (!nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_displayBL, bindingSet))
    {
        dxvk::Logger::err("ZBuffer: Failed to create binding set for display");
        return;
    }

    if (!m_displayPSO)
    {
        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_displayShader)
            .addBindingLayout(m_displayBL);

        m_displayPSO = device->createComputePipeline(computePipelineDesc);
    }

    state = nvrhi::ComputeState()
        .setPipeline(m_displayPSO)
        .addBindingSet(bindingSet);

    commandList->setComputeState(state);
    commandList->dispatch(size.x / 16 + 1, size.y / 16 + 1, 1);

    if (m_hierarchy)
    {
        m_hierarchy->Display(output, commandList);
    }
}

void ZBuffer::ReduceHierarchy(nvrhi::ICommandList* commandList)
{
    if (m_hierarchy && m_currentTexture)
    {
        nvrhi::utils::ScopedMarker marker(commandList, "ZBuffer::reduceHierarchy");
        m_hierarchy->Reduce(m_currentTexture.Get(), commandList);
    }
}

void ZBuffer::Clear(nvrhi::ICommandList* commandList)
{
    if (!m_currentTexture)
        return;
    nvrhi::utils::ScopedMarker marker(commandList, "ZBuffer::clear");
    commandList->clearTextureFloat(m_currentTexture.Get(), nvrhi::AllSubresources, nvrhi::Color(std::numeric_limits<float>::max()));
    
    if (m_hierarchy)
    {
        m_hierarchy->Clear(commandList);
    }
}