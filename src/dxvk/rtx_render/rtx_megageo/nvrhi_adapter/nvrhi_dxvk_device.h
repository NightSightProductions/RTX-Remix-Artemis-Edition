/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "nvrhi_types.h"
#include "nvrhi_dxvk_buffer.h"
#include "../../rtx_resources.h"

namespace dxvk {

  // NVRHI IDevice implementation using DXVK backend
  class NvrhiDxvkDevice : public nvrhi::IDevice {
  public:
    NvrhiDxvkDevice(
      const Rc<DxvkDevice>& device,
      const Rc<DxvkContext>& context,
      const Rc<RtxContext>& rtxContext)
      : m_device(device)
      , m_context(context)
      , m_rtxContext(rtxContext)
      , m_vkDevice(device->handle())
      , m_physicalDevice(device->adapter()->handle())
    {
    }

    // IDevice interface - Buffer management
    nvrhi::BufferHandle createBuffer(const nvrhi::BufferDesc& desc);
    void* mapBuffer(nvrhi::IBuffer* buffer, nvrhi::CpuAccessMode access);
    void unmapBuffer(nvrhi::IBuffer* buffer);

    // IDevice interface - Texture management (stub implementation)
    nvrhi::TextureHandle createTexture(const nvrhi::TextureDesc& desc);

    // IDevice interface - Sampler management
    nvrhi::SamplerHandle createSampler(const nvrhi::SamplerDesc& desc);

    // IDevice interface - Shader/Pipeline (minimal implementation)
    nvrhi::ShaderHandle createShader(
      const nvrhi::ShaderDesc& desc,
      const void* binary,
      size_t size);

    nvrhi::ComputePipelineHandle createComputePipeline(
      const nvrhi::ComputePipelineDesc& desc);

    // IDevice interface - Binding layouts (minimal implementation)
    nvrhi::BindingLayoutHandle createBindingLayout(
      const nvrhi::BindingLayoutDesc& desc);

    nvrhi::BindlessLayoutHandle createBindlessLayout(
      const nvrhi::BindlessLayoutDesc& desc);

    nvrhi::BindingSetHandle createBindingSet(
      const nvrhi::BindingSetDesc& desc,
      nvrhi::IBindingLayout* layout);

    // Create a descriptor table (empty binding set) for bindless resources.
    // This is used to satisfy pipeline binding requirements when bindless
    // layouts are included but no actual bindless descriptors are needed yet.
    nvrhi::BindingSetHandle createDescriptorTable(nvrhi::IBindingLayout* layout);

    // IDevice interface - Cluster operations (KEY FEATURE)
    nvrhi::rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(
      const nvrhi::rt::cluster::OperationParams& params);

    // IDevice interface - Command list creation
    nvrhi::CommandListHandle createCommandList(const nvrhi::CommandListParameters& params = {});

    // IDevice interface - Command list execution and synchronization
    uint64_t executeCommandList(nvrhi::ICommandList* commandList);
    void waitForIdle();

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      if (type == nvrhi::ObjectType::VK_Device) {
        nvrhi::Object obj;
        obj.pointer = (void*)(uintptr_t)m_vkDevice;
        obj.type = nvrhi::ObjectType::VK_Device;
        return obj;
      }
      if (type == nvrhi::ObjectType::VK_PhysicalDevice) {
        nvrhi::Object obj;
        obj.pointer = (void*)(uintptr_t)m_physicalDevice;
        obj.type = nvrhi::ObjectType::VK_PhysicalDevice;
        return obj;
      }
      return nvrhi::Object();
    }

    // Adapter-specific methods
    VkDevice getVkDevice() const { return m_vkDevice; }
    const Rc<DxvkDevice>& getDxvkDevice() const { return m_device; }
    const Rc<DxvkContext>& getDxvkContext() const { return m_context; }
    const Rc<RtxContext>& getRtxContext() const { return m_rtxContext; }

  private:
    Rc<DxvkDevice> m_device;
    Rc<DxvkContext> m_context;
    Rc<RtxContext> m_rtxContext;
    VkDevice m_vkDevice;
    VkPhysicalDevice m_physicalDevice;
  };

} // namespace dxvk
