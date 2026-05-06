/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <array>
#include <cstddef>
#include <vector>

#include <windows.h>

#include "../dxvk_context.h"

#include "rtx_denoise.h"

typedef struct OIDNDeviceImpl* OIDNDevice;
typedef struct OIDNFilterImpl* OIDNFilter;
typedef struct OIDNBufferImpl* OIDNBuffer;

typedef enum {
  OIDN_DEVICE_TYPE_DEFAULT = 0,
  OIDN_DEVICE_TYPE_CPU = 1,
  OIDN_DEVICE_TYPE_SYCL = 2,
  OIDN_DEVICE_TYPE_CUDA = 3,
  OIDN_DEVICE_TYPE_HIP = 4,
  OIDN_DEVICE_TYPE_METAL = 5,
} OIDNDeviceType;

typedef enum {
  OIDN_ERROR_NONE = 0,
  OIDN_ERROR_UNKNOWN = 1,
  OIDN_ERROR_INVALID_ARGUMENT = 2,
  OIDN_ERROR_INVALID_OPERATION = 3,
  OIDN_ERROR_OUT_OF_MEMORY = 4,
  OIDN_ERROR_UNSUPPORTED_HARDWARE = 5,
  OIDN_ERROR_CANCELLED = 6,
} OIDNError;

typedef enum {
  OIDN_FORMAT_UNDEFINED = 0,
  OIDN_FORMAT_FLOAT = 1,
  OIDN_FORMAT_FLOAT2 = 2,
  OIDN_FORMAT_FLOAT3 = 3,
  OIDN_FORMAT_FLOAT4 = 4,
  OIDN_FORMAT_HALF = 257,
  OIDN_FORMAT_HALF2 = 258,
  OIDN_FORMAT_HALF3 = 259,
  OIDN_FORMAT_HALF4 = 260,
} OIDNFormat;

typedef enum {
  OIDN_STORAGE_UNDEFINED = 0,
  OIDN_STORAGE_HOST = 1,
  OIDN_STORAGE_DEVICE = 2,
  OIDN_STORAGE_MANAGED = 3,
} OIDNStorage;

typedef enum {
  OIDN_QUALITY_DEFAULT = 0,
  OIDN_QUALITY_FAST = 4,
  OIDN_QUALITY_BALANCED = 5,
  OIDN_QUALITY_HIGH = 6,
} OIDNQuality;

typedef enum {
  OIDN_EXTERNAL_MEMORY_TYPE_FLAG_NONE = 0,
  OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32 = 1 << 2,
  OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT = 1 << 3,
} OIDNExternalMemoryTypeFlag;

namespace dxvk {

  class DxvkDevice;

  class OIDNContext : public CommonDeviceObject {

  public:
    OIDNContext(DxvkDevice* device, DenoiserType type);
    ~OIDNContext();

    void dispatch(
      Rc<DxvkContext> ctx,
      DxvkBarrierSet& barriers,
      const Resources::RaytracingOutput& rtOutput,
      const DxvkDenoise::Input& inputs,
      const DxvkDenoise::Output& outputs);

    void release();

    bool isAvailable() const {
      return m_isAvailable;
    }

  private:
    struct StagingImage {
      Rc<DxvkImage> image = nullptr;
      VkExtent3D extent = { 0u, 0u, 0u };
      VkFormat format = VK_FORMAT_UNDEFINED;
    };

    struct FilterTarget {
      const Resources::Resource* input = nullptr;
      const Resources::Resource* output = nullptr;
      uint32_t slot = 0;
      OIDNFormat format = OIDN_FORMAT_UNDEFINED;
      size_t pixelStride = 0;
      bool supported = false;
      bool filtered = false;
    };

    struct TemporalHistory {
      std::vector<uint8_t> data;
      uint32_t width = 0;
      uint32_t height = 0;
      size_t rowPitch = 0;
      size_t pixelStride = 0;
      OIDNFormat format = OIDN_FORMAT_UNDEFINED;
      bool valid = false;
    };

    struct InteropBuffer {
      Rc<DxvkBuffer> buffer = nullptr;
      uint32_t width = 0;
      uint32_t height = 0;
      VkFormat format = VK_FORMAT_UNDEFINED;
      size_t rowPitch = 0;
      size_t pixelStride = 0;
    };

    bool getSupportedOidnFormat(
      VkFormat vkFormat,
      OIDNFormat& outFormat,
      size_t& outPixelStride) const;

    bool ensureStagingImage(
      Rc<DxvkContext> ctx,
      uint32_t slot,
      const Resources::Resource& resource);

    void copyImage(
      Rc<DxvkContext> ctx,
      const Rc<DxvkImage>& dst,
      const Rc<DxvkImage>& src,
      const VkExtent3D& extent) const;

    bool runOidnFilter(
      const StagingImage* pStagingImage,
      const InteropBuffer* pInteropBuffer,
      size_t byteOffset,
      uint8_t* pData,
      uint32_t width,
      uint32_t height,
      size_t rowPitch,
      OIDNFormat format,
      size_t pixelStride) const;

    bool ensureInteropBuffer(
      uint32_t slot,
      uint32_t width,
      uint32_t height,
      VkFormat format,
      size_t pixelStride);

    bool ensureOidnFilterResources(
      uint32_t width,
      uint32_t height,
      size_t rowPitch,
      OIDNFormat format,
      size_t pixelStride,
      const StagingImage* pStagingImage,
      const InteropBuffer* pInteropBuffer,
      size_t byteOffset,
      bool useExternalMemory) const;

    bool getMotionVectorPixelStride(
      VkFormat format,
      size_t& outPixelStride) const;

    bool getMotionVectorXY(
      const uint8_t* pPixel,
      VkFormat format,
      float& outX,
      float& outY) const;

    void applyTemporalStabilization(
      uint32_t slot,
      uint8_t* pColorData,
      uint32_t width,
      uint32_t height,
      size_t rowPitch,
      OIDNFormat format,
      size_t pixelStride,
      const uint8_t* pMotionData,
      size_t motionRowPitch,
      size_t motionPixelStride,
      VkFormat motionFormat,
      bool resetHistory);

    void resetTemporalHistory();

    void releaseOidnFilterResources();

    OIDNQuality getFilterQuality() const;

    bool checkDeviceError(const char* operationName) const;

    DenoiserType m_type;

    mutable bool m_isAvailable = false;
    mutable bool m_loggedUnsupportedFormat = false;
    mutable bool m_loggedExternalMemoryFallback = false;
    mutable bool m_supportsExternalMemoryInterop = false;
    int m_oidnExternalMemoryTypes = 0;
    OIDNExternalMemoryTypeFlag m_oidnExternalMemoryType = OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
    VkExternalMemoryHandleTypeFlagBits m_vkExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HMODULE m_hOidn = nullptr;
    OIDNDevice m_oidnDevice = nullptr;

    Rc<sync::Fence> m_syncSignal = nullptr;
    uint64_t m_syncValue = 0;

    std::array<StagingImage, 3> m_stagingImages;
    std::array<TemporalHistory, 2> m_temporalHistory;
    std::array<InteropBuffer, 2> m_interopBuffers;

    OIDNBuffer m_oidnColorBuffer = nullptr;
    OIDNFilter m_oidnFilter = nullptr;
    uint32_t m_oidnFilterWidth = 0;
    uint32_t m_oidnFilterHeight = 0;
    size_t m_oidnFilterRowPitch = 0;
    size_t m_oidnFilterPixelStride = 0;
    OIDNFormat m_oidnFilterFormat = OIDN_FORMAT_UNDEFINED;
    OIDNQuality m_oidnFilterQuality = OIDN_QUALITY_DEFAULT;
    bool m_oidnFilterUsesExternalBuffer = false;
    VkImage m_oidnFilterExternalImage = VK_NULL_HANDLE;
    VkBuffer m_oidnFilterExternalBuffer = VK_NULL_HANDLE;
    size_t m_oidnFilterByteOffset = 0;

    mutable std::vector<uint8_t> m_oidnPackedColorBuffer;
  };

}  // namespace dxvk
