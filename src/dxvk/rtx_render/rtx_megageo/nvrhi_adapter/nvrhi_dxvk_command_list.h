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

#include <memory>
#include "nvrhi_types.h"
#include "nvrhi_dxvk_device.h"
#include "nvrhi_scratch_manager.h"

namespace dxvk {

  // NVRHI ICommandList implementation using DxvkContext
  class NvrhiDxvkCommandList : public nvrhi::ICommandList {
  public:
    // Default scratch chunk size (256MB - large enough for most cluster operations)
    static constexpr uint64_t kDefaultScratchChunkSize = 256 * 1024 * 1024;

    NvrhiDxvkCommandList(
      NvrhiDxvkDevice* device,
      const Rc<DxvkContext>& context)
      : m_device(device)
      , m_context(context)
      , m_scratchManager(std::make_unique<ScratchManager>(device->getDxvkDevice().ptr(), kDefaultScratchChunkSize))
    {
    }

    // ICommandList interface - Lifecycle
    void open();
    void close();
    void clearState();

    // ICommandList interface - Buffer operations
    void writeBuffer(
      nvrhi::IBuffer* buffer,
      const void* data,
      size_t size,
      uint64_t offset = 0);

    void clearBufferUInt(
      nvrhi::IBuffer* buffer,
      uint32_t value);

    void copyBuffer(
      nvrhi::IBuffer* dst,
      uint64_t dstOffset,
      nvrhi::IBuffer* src,
      uint64_t srcOffset,
      uint64_t size);

    // ICommandList interface - Texture operations
    void copyTexture(nvrhi::ITexture* dst, nvrhi::ITexture* src);
    void clearTextureFloat(nvrhi::ITexture* texture, const nvrhi::TextureSubresourceSet& subresources, const nvrhi::Color& clearColor);

    // ICommandList interface - Compute
    void setComputeState(const nvrhi::ComputeState& state);
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void dispatchIndirect(nvrhi::IBuffer* buffer, uint64_t offset);
    void dispatchIndirect(uint64_t offset);  // Uses buffer from current compute state

    // ICommandList interface - Cluster operations (KEY FEATURE)
    void executeMultiIndirectClusterOperation(
      const nvrhi::rt::cluster::OperationDesc& desc);

    // ICommandList interface - Barriers
    void bufferBarrier(
      nvrhi::IBuffer* buffer,
      nvrhi::ResourceStates stateBefore,
      nvrhi::ResourceStates stateAfter);

    void globalBarrier(
      nvrhi::ResourceStates stateBefore,
      nvrhi::ResourceStates stateAfter);

    // Device access
    nvrhi::IDevice* getDevice() override {
      return m_device;
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      if (type == nvrhi::ObjectType::VK_CommandBuffer) {
        nvrhi::Object obj;
        obj.pointer = (void*)(uintptr_t)m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
        obj.type = nvrhi::ObjectType::VK_CommandBuffer;
        return obj;
      }
      return nvrhi::Object();
    }

  private:
    NvrhiDxvkDevice* m_device;
    Rc<DxvkContext> m_context;
    nvrhi::ComputeState m_computeState;
    std::unique_ptr<ScratchManager> m_scratchManager;

    void bindComputeResources(const nvrhi::ComputeState& state);
    void translateClusterOperation(
      const nvrhi::rt::cluster::OperationDesc& nvrhiDesc,
      VkClusterAccelerationStructureCommandsInfoNV& vkCmds);
  };

} // namespace dxvk
