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
#include "../../../util/log/log.h"

namespace dxvk {

  // NVRHI IBuffer implementation wrapping DxvkBuffer
  class NvrhiDxvkBuffer : public nvrhi::IBuffer {
  public:
    NvrhiDxvkBuffer(
      const nvrhi::BufferDesc& desc,
      const Rc<DxvkBuffer>& dxvkBuffer)
      : m_desc(desc)
      , m_dxvkBuffer(dxvkBuffer)
    {
    }

    // IBuffer interface
    const nvrhi::BufferDesc& getDesc() const { return m_desc; }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      if (type == nvrhi::ObjectType::VK_Buffer) {
        nvrhi::Object obj;
        obj.pointer = (void*)(uintptr_t)m_dxvkBuffer->getBufferRaw();
        obj.type = nvrhi::ObjectType::VK_Buffer;
        return obj;
      }
      return nvrhi::Object();
    }

    nvrhi::GpuVirtualAddress getGpuVirtualAddress() const {
      // Get Vulkan buffer device address
      return m_dxvkBuffer->getDeviceAddress();
    }

    // Adapter-specific methods
    const Rc<DxvkBuffer>& getDxvkBuffer() const { return m_dxvkBuffer; }

    DxvkBufferSliceHandle getSliceHandle() const {
      return m_dxvkBuffer->getSliceHandle();
    }

    DxvkBufferSliceHandle getSliceHandle(VkDeviceSize offset, VkDeviceSize length) const {
      return m_dxvkBuffer->getSliceHandle(offset, length);
    }

    // Cache data for push constants (used for constant buffers < 128 bytes)
    void setCachedData(const void* data, size_t size) {
      if (size <= sizeof(m_cachedData)) {
        memcpy(m_cachedData, data, size);
        m_cachedDataSize = size;
      }
    }

    const void* getCachedData() const { return m_cachedData; }
    size_t getCachedDataSize() const { return m_cachedDataSize; }
    bool hasCachedData() const { return m_cachedDataSize > 0; }

  private:
    nvrhi::BufferDesc m_desc;
    Rc<DxvkBuffer> m_dxvkBuffer;

    // Cache for push constant data (max 128 bytes)
    uint8_t m_cachedData[128] = {};
    size_t m_cachedDataSize = 0;
  };

} // namespace dxvk
