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
#include "rtx_oidn_context.h"

#include "../dxvk_buffer.h"
#include "../dxvk_device.h"
#include "../../util/util_string.h"

#include "rtx_options.h"

#include <algorithm>
#include <filesystem>
#include <array>
#include <cmath>
#include <cstring>

#include <glm/gtc/packing.hpp>

namespace oidn {
  using pfnNewDevice = OIDNDevice(*)(OIDNDeviceType type);
  using pfnCommitDevice = void (*)(OIDNDevice device);
  using pfnReleaseDevice = void (*)(OIDNDevice device);
  using pfnGetDeviceError = OIDNError (*)(OIDNDevice device, const char** outMessage);
  using pfnGetDeviceInt = int (*)(OIDNDevice device, const char* name);
  using pfnNewBufferWithStorage = OIDNBuffer (*)(OIDNDevice device, size_t byteSize, OIDNStorage storage);
  using pfnNewSharedBufferFromWin32Handle = OIDNBuffer (*)(OIDNDevice device,
                                                            OIDNExternalMemoryTypeFlag handleType,
                                                            void* handle,
                                                            const void* name,
                                                            size_t byteSize);
  using pfnReleaseBuffer = void (*)(OIDNBuffer buffer);
  using pfnWriteBuffer = void (*)(OIDNBuffer buffer, size_t byteOffset, size_t byteSize, const void* srcHostPtr);
  using pfnReadBuffer = void (*)(OIDNBuffer buffer, size_t byteOffset, size_t byteSize, void* dstHostPtr);
  using pfnNewFilter = OIDNFilter(*)(OIDNDevice device, const char* type);
  using pfnReleaseFilter = void (*)(OIDNFilter filter);
  using pfnSetFilterImage = void (*)(OIDNFilter filter, const char* name,
                                     OIDNBuffer buffer, OIDNFormat format,
                                     size_t width, size_t height,
                                     size_t byteOffset,
                                     size_t pixelByteStride, size_t rowByteStride);
  using pfnSetFilterBool = void (*)(OIDNFilter filter, const char* name, bool value);
  using pfnSetFilterInt = void (*)(OIDNFilter filter, const char* name, int value);
  using pfnCommitFilter = void (*)(OIDNFilter filter);
  using pfnExecuteFilter = void (*)(OIDNFilter filter);

  struct DispatchOIDN {
    pfnNewDevice NewDevice;
    pfnCommitDevice CommitDevice;
    pfnReleaseDevice ReleaseDevice;
    pfnGetDeviceError GetDeviceError;
    pfnGetDeviceInt GetDeviceInt;
    pfnNewBufferWithStorage NewBufferWithStorage;
    pfnNewSharedBufferFromWin32Handle NewSharedBufferFromWin32Handle;
    pfnReleaseBuffer ReleaseBuffer;
    pfnWriteBuffer WriteBuffer;
    pfnReadBuffer ReadBuffer;
    pfnNewFilter NewFilter;
    pfnReleaseFilter ReleaseFilter;
    pfnSetFilterImage SetFilterImage;
    pfnSetFilterBool SetFilterBool;
    pfnSetFilterInt SetFilterInt;
    pfnCommitFilter CommitFilter;
    pfnExecuteFilter ExecuteFilter;
  } dispatch;

  HMODULE initialize() {
    HMODULE hModule;
    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCTSTR>(&initialize),
      &hModule);

    wchar_t modulePath[MAX_PATH];
    GetModuleFileNameW(hModule, modulePath, MAX_PATH);

    std::filesystem::path oidnPath(modulePath);
    oidnPath = oidnPath.parent_path() / L"OpenImageDenoise.dll";

    HMODULE hOidn = LoadLibraryW(oidnPath.c_str());
    if (hOidn == nullptr) {
      dxvk::Logger::warn("[RTX] OIDN: OpenImageDenoise.dll not found, backend disabled");
      return nullptr;
    }

#define GET_OIDN_PROC(proc) dispatch.proc = (pfn##proc)GetProcAddress(hOidn, "oidn" #proc)
    GET_OIDN_PROC(NewDevice);
    GET_OIDN_PROC(CommitDevice);
    GET_OIDN_PROC(ReleaseDevice);
    GET_OIDN_PROC(GetDeviceError);
    GET_OIDN_PROC(GetDeviceInt);
    GET_OIDN_PROC(NewBufferWithStorage);
    GET_OIDN_PROC(NewSharedBufferFromWin32Handle);
    GET_OIDN_PROC(ReleaseBuffer);
    GET_OIDN_PROC(WriteBuffer);
    GET_OIDN_PROC(ReadBuffer);
    GET_OIDN_PROC(NewFilter);
    GET_OIDN_PROC(ReleaseFilter);
    GET_OIDN_PROC(SetFilterImage);
    GET_OIDN_PROC(SetFilterBool);
    GET_OIDN_PROC(SetFilterInt);
    GET_OIDN_PROC(CommitFilter);
    GET_OIDN_PROC(ExecuteFilter);
#undef GET_OIDN_PROC

    if (!dispatch.NewDevice || !dispatch.CommitDevice || !dispatch.ReleaseDevice
      || !dispatch.GetDeviceError || !dispatch.GetDeviceInt || !dispatch.NewBufferWithStorage
      || !dispatch.ReleaseBuffer || !dispatch.WriteBuffer || !dispatch.ReadBuffer
      || !dispatch.NewFilter || !dispatch.ReleaseFilter || !dispatch.SetFilterImage
      || !dispatch.SetFilterBool || !dispatch.SetFilterInt
     || !dispatch.CommitFilter || !dispatch.ExecuteFilter) {
      dxvk::Logger::err("[RTX] OIDN: Failed to resolve OIDN exports");
      FreeLibrary(hOidn);
      return nullptr;
    }

    return hOidn;
  }
}

namespace dxvk {

  OIDNContext::OIDNContext(DxvkDevice* device, DenoiserType type)
    : CommonDeviceObject(device), m_type(type) {
    m_hOidn = oidn::initialize();
    if (m_hOidn == nullptr) {
      return;
    }

    static const std::array<OIDNDeviceType, 4> kPreferredGpuBackends = {
      OIDN_DEVICE_TYPE_DEFAULT,
      OIDN_DEVICE_TYPE_SYCL,
      OIDN_DEVICE_TYPE_HIP,
      OIDN_DEVICE_TYPE_CUDA,
    };

    int selectedDeviceType = -1;

    for (OIDNDeviceType backendType : kPreferredGpuBackends) {
      OIDNDevice candidate = oidn::dispatch.NewDevice(backendType);
      if (candidate == nullptr) {
        continue;
      }

      oidn::dispatch.CommitDevice(candidate);

      const OIDNError error = oidn::dispatch.GetDeviceError(candidate, nullptr);
      if (error != OIDN_ERROR_NONE) {
        oidn::dispatch.ReleaseDevice(candidate);
        continue;
      }

      const int deviceType = oidn::dispatch.GetDeviceInt(candidate, "type");
      const bool isGpuDevice =
        deviceType == static_cast<int>(OIDN_DEVICE_TYPE_CUDA)
        || deviceType == static_cast<int>(OIDN_DEVICE_TYPE_HIP)
        || deviceType == static_cast<int>(OIDN_DEVICE_TYPE_SYCL)
        || deviceType == static_cast<int>(OIDN_DEVICE_TYPE_METAL);

      if (!isGpuDevice) {
        oidn::dispatch.ReleaseDevice(candidate);
        continue;
      }

      m_oidnDevice = candidate;
      selectedDeviceType = deviceType;
      break;
    }

    if (m_oidnDevice == nullptr) {
      Logger::warn("[RTX] OIDN: No compatible GPU backend found, backend disabled");
      return;
    }

    int externalMemoryTypes = 0;
    if (oidn::dispatch.GetDeviceInt != nullptr) {
      externalMemoryTypes = oidn::dispatch.GetDeviceInt(m_oidnDevice, "externalMemoryTypes");
      if (!checkDeviceError("querying external memory support")) {
        externalMemoryTypes = 0;
      }
    }

    m_oidnExternalMemoryTypes = externalMemoryTypes;

    if (oidn::dispatch.NewSharedBufferFromWin32Handle != nullptr) {
      if ((externalMemoryTypes & OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) != 0) {
        m_oidnExternalMemoryType = OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
        m_vkExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        m_supportsExternalMemoryInterop = true;
      } else if ((externalMemoryTypes & OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT) != 0) {
        m_oidnExternalMemoryType = OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT;
        m_vkExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
        m_supportsExternalMemoryInterop = true;
      }
    }

    Logger::info(str::format("[RTX] OIDN: Initialized GPU backend type ", selectedDeviceType));
    Logger::info(str::format(
      "[RTX] OIDN: External Win32 memory interop ",
      m_supportsExternalMemoryInterop ? "enabled" : "unavailable, GPU-only OIDN path disabled"));
    if (m_supportsExternalMemoryInterop) {
      Logger::info(str::format(
        "[RTX] OIDN: External memory handle type ",
        m_oidnExternalMemoryType == OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32 ? "OPAQUE_WIN32" : "OPAQUE_WIN32_KMT"));
    }

    m_syncSignal = new sync::Fence(m_syncValue);
    m_isAvailable = true;
  }

  OIDNContext::~OIDNContext() {
    release();

    if (m_oidnDevice != nullptr) {
      oidn::dispatch.ReleaseDevice(m_oidnDevice);
      m_oidnDevice = nullptr;
    }

    if (m_hOidn != nullptr) {
      FreeLibrary(m_hOidn);
      m_hOidn = nullptr;
    }
  }

  void OIDNContext::release() {
    releaseOidnFilterResources();
    resetTemporalHistory();
    m_oidnPackedColorBuffer.clear();
    m_loggedExternalMemoryFallback = false;

    for (auto& interop : m_interopBuffers) {
      interop.buffer = nullptr;
      interop.width = 0;
      interop.height = 0;
      interop.format = VK_FORMAT_UNDEFINED;
      interop.rowPitch = 0;
      interop.pixelStride = 0;
    }

    for (auto& image : m_stagingImages) {
      image.image = nullptr;
      image.extent = { 0u, 0u, 0u };
      image.format = VK_FORMAT_UNDEFINED;
    }
  }

  void OIDNContext::releaseOidnFilterResources() {
    if (m_oidnFilter != nullptr) {
      oidn::dispatch.ReleaseFilter(m_oidnFilter);
      m_oidnFilter = nullptr;
    }

    if (m_oidnColorBuffer != nullptr) {
      oidn::dispatch.ReleaseBuffer(m_oidnColorBuffer);
      m_oidnColorBuffer = nullptr;
    }

    m_oidnFilterWidth = 0;
    m_oidnFilterHeight = 0;
    m_oidnFilterRowPitch = 0;
    m_oidnFilterPixelStride = 0;
    m_oidnFilterFormat = OIDN_FORMAT_UNDEFINED;
    m_oidnFilterQuality = OIDN_QUALITY_DEFAULT;
    m_oidnFilterUsesExternalBuffer = false;
    m_oidnFilterExternalImage = VK_NULL_HANDLE;
    m_oidnFilterExternalBuffer = VK_NULL_HANDLE;
    m_oidnFilterByteOffset = 0;
  }

  void OIDNContext::resetTemporalHistory() {
    for (auto& history : m_temporalHistory) {
      history.data.clear();
      history.width = 0;
      history.height = 0;
      history.rowPitch = 0;
      history.pixelStride = 0;
      history.format = OIDN_FORMAT_UNDEFINED;
      history.valid = false;
    }
  }

  bool OIDNContext::getMotionVectorPixelStride(
    VkFormat format,
    size_t& outPixelStride) const {
    switch (format) {
    case VK_FORMAT_R16G16_SFLOAT:
      outPixelStride = sizeof(uint16_t) * 2;
      return true;

    case VK_FORMAT_R16G16B16A16_SFLOAT:
      outPixelStride = sizeof(uint16_t) * 4;
      return true;

    case VK_FORMAT_R32G32_SFLOAT:
      outPixelStride = sizeof(float) * 2;
      return true;

    case VK_FORMAT_R32G32B32A32_SFLOAT:
      outPixelStride = sizeof(float) * 4;
      return true;

    default:
      outPixelStride = 0;
      return false;
    }
  }

  bool OIDNContext::getMotionVectorXY(
    const uint8_t* pPixel,
    VkFormat format,
    float& outX,
    float& outY) const {
    if (pPixel == nullptr) {
      outX = 0.f;
      outY = 0.f;
      return false;
    }

    switch (format) {
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
      uint32_t packed = 0;
      std::memcpy(&packed, pPixel, sizeof(uint32_t));
      const uint16_t xHalf = static_cast<uint16_t>(packed & 0xFFFFu);
      const uint16_t yHalf = static_cast<uint16_t>((packed >> 16) & 0xFFFFu);
      outX = glm::unpackHalf1x16(xHalf);
      outY = glm::unpackHalf1x16(yHalf);
      return true;
    }

    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      std::memcpy(&outX, pPixel + 0, sizeof(float));
      std::memcpy(&outY, pPixel + sizeof(float), sizeof(float));
      return true;

    default:
      outX = 0.f;
      outY = 0.f;
      return false;
    }
  }

  void OIDNContext::applyTemporalStabilization(
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
    bool resetHistory) {
    if (slot >= m_temporalHistory.size() || pColorData == nullptr || width == 0 || height == 0) {
      return;
    }

    TemporalHistory& history = m_temporalHistory[slot];
    const size_t byteSize = rowPitch * static_cast<size_t>(height);

    const bool historyCompatible =
      history.valid
      && history.width == width
      && history.height == height
      && history.rowPitch == rowPitch
      && history.pixelStride == pixelStride
      && history.format == format
      && history.data.size() == byteSize;

    const bool motionValid =
      pMotionData != nullptr
      && motionRowPitch >= motionPixelStride * static_cast<size_t>(width)
      && motionPixelStride > 0;

    if (resetHistory || !historyCompatible || !motionValid) {
      history.data.assign(pColorData, pColorData + byteSize);
      history.width = width;
      history.height = height;
      history.rowPitch = rowPitch;
      history.pixelStride = pixelStride;
      history.format = format;
      history.valid = true;
      return;
    }

    auto blendHalf3Pixel = [](uint8_t* pCurrent, const uint8_t* pPrevious, float historyWeight) {
      for (size_t c = 0; c < 3; ++c) {
        uint16_t currentHalf = 0;
        uint16_t previousHalf = 0;

        std::memcpy(&currentHalf, pCurrent + c * sizeof(uint16_t), sizeof(uint16_t));
        std::memcpy(&previousHalf, pPrevious + c * sizeof(uint16_t), sizeof(uint16_t));

        const float currentValue = glm::unpackHalf1x16(currentHalf);
        const float previousValue = glm::unpackHalf1x16(previousHalf);
        const float blended = currentValue * (1.0f - historyWeight) + previousValue * historyWeight;
        const uint16_t blendedHalf = glm::packHalf1x16(blended);

        std::memcpy(pCurrent + c * sizeof(uint16_t), &blendedHalf, sizeof(uint16_t));
      }
    };

    auto blendFloat3Pixel = [](uint8_t* pCurrent, const uint8_t* pPrevious, float historyWeight) {
      for (size_t c = 0; c < 3; ++c) {
        float currentValue = 0.f;
        float previousValue = 0.f;

        std::memcpy(&currentValue, pCurrent + c * sizeof(float), sizeof(float));
        std::memcpy(&previousValue, pPrevious + c * sizeof(float), sizeof(float));

        const float blended = currentValue * (1.0f - historyWeight) + previousValue * historyWeight;
        std::memcpy(pCurrent + c * sizeof(float), &blended, sizeof(float));
      }
    };

    for (uint32_t y = 0; y < height; ++y) {
      uint8_t* pColorRow = pColorData + rowPitch * static_cast<size_t>(y);
      const uint8_t* pMotionRow = pMotionData + motionRowPitch * static_cast<size_t>(y);

      for (uint32_t x = 0; x < width; ++x) {
        float motionX = 0.f;
        float motionY = 0.f;
        const uint8_t* pMotionPixel = pMotionRow + motionPixelStride * static_cast<size_t>(x);

        if (!getMotionVectorXY(pMotionPixel, motionFormat, motionX, motionY)
         || !std::isfinite(motionX)
         || !std::isfinite(motionY)) {
          continue;
        }

        const int prevX = static_cast<int>(std::lround(static_cast<float>(x) + motionX));
        const int prevY = static_cast<int>(std::lround(static_cast<float>(y) + motionY));
        if (prevX < 0 || prevY < 0 || prevX >= static_cast<int>(width) || prevY >= static_cast<int>(height)) {
          continue;
        }

        const float motionLength = std::sqrt(motionX * motionX + motionY * motionY);
        const float historyWeight = std::clamp(0.82f - motionLength * 0.35f, 0.0f, 0.82f);
        if (historyWeight <= 0.f) {
          continue;
        }

        uint8_t* pCurrentPixel = pColorRow + pixelStride * static_cast<size_t>(x);
        const uint8_t* pPreviousPixel =
          history.data.data()
          + history.rowPitch * static_cast<size_t>(prevY)
          + history.pixelStride * static_cast<size_t>(prevX);

        if (format == OIDN_FORMAT_HALF3) {
          blendHalf3Pixel(pCurrentPixel, pPreviousPixel, historyWeight);
        } else if (format == OIDN_FORMAT_FLOAT3) {
          blendFloat3Pixel(pCurrentPixel, pPreviousPixel, historyWeight);
        }
      }
    }

    std::memcpy(history.data.data(), pColorData, byteSize);
    history.valid = true;
  }

  bool OIDNContext::getSupportedOidnFormat(
    VkFormat vkFormat,
    OIDNFormat& outFormat,
    size_t& outPixelStride) const {
    switch (vkFormat) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      outFormat = OIDN_FORMAT_HALF3;
      outPixelStride = sizeof(uint16_t) * 4;
      return true;

    case VK_FORMAT_R32G32B32A32_SFLOAT:
      outFormat = OIDN_FORMAT_FLOAT3;
      outPixelStride = sizeof(float) * 4;
      return true;

    default:
      outFormat = OIDN_FORMAT_UNDEFINED;
      outPixelStride = 0;
      return false;
    }
  }

  bool OIDNContext::ensureStagingImage(
    Rc<DxvkContext> ctx,
    uint32_t slot,
    const Resources::Resource& resource) {
    assert(slot < m_stagingImages.size());

    StagingImage& staging = m_stagingImages[slot];
    const VkExtent3D extent = resource.image->info().extent;
    const VkFormat format = resource.image->info().format;

    if (staging.image != nullptr
     && staging.extent.width == extent.width
     && staging.extent.height == extent.height
     && staging.extent.depth == extent.depth
     && staging.format == format) {
      return true;
    }

    DxvkImageCreateInfo imageInfo = {};
    imageInfo.type = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.flags = 0;
    imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.extent = extent;
    imageInfo.extent.depth = 1;
    imageInfo.numLayers = 1;
    imageInfo.mipLevels = 1;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    imageInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.layout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (m_supportsExternalMemoryInterop) {
      imageInfo.sharing.mode = DxvkSharedHandleMode::Export;
      imageInfo.sharing.type = m_vkExternalMemoryHandleType;
    }

    VkImageFormatProperties imageFormatProperties = {};
    if (device()->adapter()->imageFormatProperties(
      imageInfo.format,
      imageInfo.type,
      imageInfo.tiling,
      imageInfo.usage,
      imageInfo.flags,
      imageFormatProperties) != VK_SUCCESS) {
      Logger::warn("[RTX] OIDN: Linear staging format unsupported, falling back to unfiltered signal");
      staging.image = nullptr;
      staging.extent = { 0u, 0u, 0u };
      staging.format = VK_FORMAT_UNDEFINED;
      return false;
    }

    staging.image = device()->createImage(
      imageInfo,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXRenderTarget,
      "oidn staging image");

    if (staging.image == nullptr) {
      staging.extent = { 0u, 0u, 0u };
      staging.format = VK_FORMAT_UNDEFINED;
      return false;
    }

    ctx->changeImageLayout(staging.image, VK_IMAGE_LAYOUT_GENERAL);

    staging.extent = imageInfo.extent;
    staging.format = imageInfo.format;
    return true;
  }

  bool OIDNContext::ensureInteropBuffer(
    uint32_t slot,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    size_t pixelStride) {
    assert(slot < m_interopBuffers.size());

    if (width == 0 || height == 0 || pixelStride == 0) {
      return false;
    }

    const size_t rowPitch = static_cast<size_t>(width) * pixelStride;
    if (rowPitch < pixelStride) {
      return false;
    }

    const size_t byteSize = rowPitch * static_cast<size_t>(height);
    if (byteSize < rowPitch) {
      return false;
    }

    InteropBuffer& interop = m_interopBuffers[slot];
    if (interop.buffer != nullptr
     && interop.width == width
     && interop.height == height
     && interop.format == format
     && interop.rowPitch == rowPitch
     && interop.pixelStride == pixelStride) {
      return true;
    }

    DxvkBufferCreateInfo bufferInfo = {};
    bufferInfo.size = byteSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    bufferInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

    if (m_supportsExternalMemoryInterop) {
      bufferInfo.sharing.mode = DxvkSharedHandleMode::Export;
      bufferInfo.sharing.type = m_vkExternalMemoryHandleType;
    }

    interop.buffer = device()->createBuffer(
      bufferInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      "oidn interop buffer");

    if (interop.buffer == nullptr) {
      interop.width = 0;
      interop.height = 0;
      interop.format = VK_FORMAT_UNDEFINED;
      interop.rowPitch = 0;
      interop.pixelStride = 0;
      return false;
    }

    interop.width = width;
    interop.height = height;
    interop.format = format;
    interop.rowPitch = rowPitch;
    interop.pixelStride = pixelStride;
    return true;
  }

  void OIDNContext::copyImage(
    Rc<DxvkContext> ctx,
    const Rc<DxvkImage>& dst,
    const Rc<DxvkImage>& src,
    const VkExtent3D& extent) const {
    if (dst == nullptr || src == nullptr || dst == src) {
      return;
    }

    VkImageSubresourceLayers subresource = {};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel = 0;
    subresource.baseArrayLayer = 0;
    subresource.layerCount = 1;

    ctx->copyImage(
      dst,
      subresource,
      VkOffset3D{ 0, 0, 0 },
      src,
      subresource,
      VkOffset3D{ 0, 0, 0 },
      extent);
  }

  OIDNQuality OIDNContext::getFilterQuality() const {
    switch (RtxOptions::oidnQuality()) {
    case OidnQuality::Fast:
      return OIDN_QUALITY_FAST;
    case OidnQuality::High:
      return OIDN_QUALITY_HIGH;
    case OidnQuality::Balanced:
    default:
      return OIDN_QUALITY_BALANCED;
    }
  }

  bool OIDNContext::checkDeviceError(const char* operationName) const {
    if (m_oidnDevice == nullptr) {
      return false;
    }

    const char* errorMessage = nullptr;
    const OIDNError error = oidn::dispatch.GetDeviceError(m_oidnDevice, &errorMessage);
    if (error == OIDN_ERROR_NONE) {
      return true;
    }

    Logger::err(str::format("[RTX] OIDN: Error while ", operationName, ": ", static_cast<int>(error), " ",
      errorMessage != nullptr ? errorMessage : "<no details>"));
    return false;
  }

  bool OIDNContext::ensureOidnFilterResources(
    uint32_t width,
    uint32_t height,
    size_t rowPitch,
    OIDNFormat format,
    size_t pixelStride,
    const OIDNContext::StagingImage* pStagingImage,
    const OIDNContext::InteropBuffer* pInteropBuffer,
    size_t byteOffset,
    bool useExternalMemory) const {
    if (m_oidnDevice == nullptr) {
      return false;
    }

    const bool hasExternalImage = pStagingImage != nullptr && pStagingImage->image != nullptr;
    const bool hasExternalBuffer = pInteropBuffer != nullptr && pInteropBuffer->buffer != nullptr;
    const bool externalMemoryPath =
      useExternalMemory
      && (hasExternalImage || hasExternalBuffer);

    const size_t externalBufferMemoryOffset = hasExternalBuffer
      ? static_cast<size_t>(pInteropBuffer->buffer->getBufferHandle().memory.offset())
      : 0;

    const size_t effectiveByteOffset = externalMemoryPath
      ? (hasExternalBuffer ? externalBufferMemoryOffset + byteOffset : byteOffset)
      : 0;

    const VkImage expectedExternalImage = hasExternalImage
      ? pStagingImage->image->handle()
      : VK_NULL_HANDLE;

    const VkBuffer expectedExternalBuffer = hasExternalBuffer
      ? pInteropBuffer->buffer->getBufferRaw()
      : VK_NULL_HANDLE;

    const OIDNQuality requestedQuality = getFilterQuality();
    const bool needsRecreate =
      m_oidnFilter == nullptr
      || m_oidnColorBuffer == nullptr
      || m_oidnFilterWidth != width
      || m_oidnFilterHeight != height
      || m_oidnFilterRowPitch != rowPitch
      || m_oidnFilterPixelStride != pixelStride
      || m_oidnFilterFormat != format
      || m_oidnFilterQuality != requestedQuality
      || m_oidnFilterUsesExternalBuffer != externalMemoryPath
      || (externalMemoryPath
        && (m_oidnFilterExternalImage != expectedExternalImage
         || m_oidnFilterExternalBuffer != expectedExternalBuffer
         || m_oidnFilterByteOffset != effectiveByteOffset));

    if (!needsRecreate) {
      return true;
    }

    // Recreate filter and backing buffer only when image layout or quality changes.
    const size_t byteSize = externalMemoryPath
      ? (hasExternalBuffer
        ? static_cast<size_t>(pInteropBuffer->buffer->getBufferHandle().memory.length())
        : static_cast<size_t>(pStagingImage->image->memSize()))
      : rowPitch * static_cast<size_t>(height);

    const_cast<OIDNContext*>(this)->releaseOidnFilterResources();

    OIDNBuffer colorBuffer = nullptr;

    if (externalMemoryPath) {
      HANDLE sharedHandle = hasExternalBuffer
        ? pInteropBuffer->buffer->sharedHandle()
        : pStagingImage->image->sharedHandle();

      if (sharedHandle == nullptr || sharedHandle == INVALID_HANDLE_VALUE) {
        return false;
      }

      colorBuffer = oidn::dispatch.NewSharedBufferFromWin32Handle(
        m_oidnDevice,
        m_oidnExternalMemoryType,
        sharedHandle,
        nullptr,
        byteSize);

      if (colorBuffer == nullptr
       && m_oidnExternalMemoryType != OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT) {
        CloseHandle(sharedHandle);
      }
    } else {
      colorBuffer = oidn::dispatch.NewBufferWithStorage(m_oidnDevice, byteSize, OIDN_STORAGE_DEVICE);
    }

    if (colorBuffer == nullptr) {
      checkDeviceError(externalMemoryPath ? "importing shared GPU buffer" : "creating GPU buffer");
      return false;
    }

    OIDNFilter filter = oidn::dispatch.NewFilter(m_oidnDevice, "RT");
    if (filter == nullptr) {
      checkDeviceError("creating filter");
      oidn::dispatch.ReleaseBuffer(colorBuffer);
      return false;
    }

    oidn::dispatch.SetFilterImage(
      filter,
      "color",
      colorBuffer,
      format,
      width,
      height,
      effectiveByteOffset,
      pixelStride,
      rowPitch);

    oidn::dispatch.SetFilterImage(
      filter,
      "output",
      colorBuffer,
      format,
      width,
      height,
      effectiveByteOffset,
      pixelStride,
      rowPitch);

    oidn::dispatch.SetFilterBool(filter, "hdr", true);
    oidn::dispatch.SetFilterInt(filter, "quality", static_cast<int>(requestedQuality));

    oidn::dispatch.CommitFilter(filter);
    if (!checkDeviceError("committing filter")) {
      oidn::dispatch.ReleaseFilter(filter);
      oidn::dispatch.ReleaseBuffer(colorBuffer);
      return false;
    }

    OIDNContext* mutableThis = const_cast<OIDNContext*>(this);
    mutableThis->m_oidnColorBuffer = colorBuffer;
    mutableThis->m_oidnFilter = filter;
    mutableThis->m_oidnFilterWidth = width;
    mutableThis->m_oidnFilterHeight = height;
    mutableThis->m_oidnFilterRowPitch = rowPitch;
    mutableThis->m_oidnFilterPixelStride = pixelStride;
    mutableThis->m_oidnFilterFormat = format;
    mutableThis->m_oidnFilterQuality = requestedQuality;
    mutableThis->m_oidnFilterUsesExternalBuffer = externalMemoryPath;
    mutableThis->m_oidnFilterExternalImage = externalMemoryPath ? expectedExternalImage : VK_NULL_HANDLE;
    mutableThis->m_oidnFilterExternalBuffer = externalMemoryPath ? expectedExternalBuffer : VK_NULL_HANDLE;
    mutableThis->m_oidnFilterByteOffset = effectiveByteOffset;

    return true;
  }

  bool OIDNContext::runOidnFilter(
    const OIDNContext::StagingImage* pStagingImage,
    const OIDNContext::InteropBuffer* pInteropBuffer,
    size_t byteOffset,
    uint8_t* pData,
    uint32_t width,
    uint32_t height,
    size_t rowPitch,
    OIDNFormat format,
    size_t pixelStride) const {
    if (m_oidnDevice == nullptr || width == 0 || height == 0) {
      return false;
    }

    const bool useExternalMemoryPath =
      m_supportsExternalMemoryInterop
      && ((pStagingImage != nullptr && pStagingImage->image != nullptr)
       || (pInteropBuffer != nullptr && pInteropBuffer->buffer != nullptr));

    if (useExternalMemoryPath) {
      if (ensureOidnFilterResources(
        width,
        height,
        rowPitch,
        format,
        pixelStride,
        pStagingImage,
        pInteropBuffer,
        byteOffset,
        true)) {
        oidn::dispatch.ExecuteFilter(m_oidnFilter);
        if (checkDeviceError("executing filter")) {
          return true;
        }

        OIDNContext* mutableThis = const_cast<OIDNContext*>(this);

        if (!m_loggedExternalMemoryFallback) {
          Logger::warn("[RTX] OIDN: Shared-memory interop execution failed, switching to CPU transfer fallback");
          mutableThis->m_loggedExternalMemoryFallback = true;
        }

        mutableThis->m_supportsExternalMemoryInterop = false;
        mutableThis->releaseOidnFilterResources();
      } else if (m_supportsExternalMemoryInterop) {
        OIDNContext* mutableThis = const_cast<OIDNContext*>(this);

        const bool canRetryWithKmt =
          mutableThis->m_oidnExternalMemoryType == OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32
          && (mutableThis->m_oidnExternalMemoryTypes & OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT) != 0;

        if (canRetryWithKmt) {
          Logger::warn("[RTX] OIDN: Retrying shared-memory interop with OPAQUE_WIN32_KMT handle type");
          mutableThis->m_oidnExternalMemoryType = OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT;
          mutableThis->m_vkExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
          mutableThis->releaseOidnFilterResources();

          for (auto& image : mutableThis->m_stagingImages) {
            image.image = nullptr;
            image.extent = { 0u, 0u, 0u };
            image.format = VK_FORMAT_UNDEFINED;
          }

          for (auto& interop : mutableThis->m_interopBuffers) {
            interop.buffer = nullptr;
            interop.width = 0;
            interop.height = 0;
            interop.format = VK_FORMAT_UNDEFINED;
            interop.rowPitch = 0;
            interop.pixelStride = 0;
          }

          return false;
        }

        if (!m_loggedExternalMemoryFallback) {
          Logger::warn("[RTX] OIDN: Shared-memory interop unavailable, switching to CPU transfer fallback");
          mutableThis->m_loggedExternalMemoryFallback = true;
        }

        mutableThis->m_supportsExternalMemoryInterop = false;
        mutableThis->releaseOidnFilterResources();
      }
    }

    if (pData == nullptr && pStagingImage != nullptr && pStagingImage->image != nullptr) {
      pData = reinterpret_cast<uint8_t*>(pStagingImage->image->mapPtr(byteOffset));
    }

    if (pData == nullptr) {
      return false;
    }

    const size_t tightRowPitch = static_cast<size_t>(width) * pixelStride;
    if (tightRowPitch == 0 || rowPitch < tightRowPitch) {
      return false;
    }

    uint8_t* pFilterData = pData;
    size_t filterRowPitch = rowPitch;

    if (rowPitch != tightRowPitch) {
      const size_t packedByteSize = tightRowPitch * static_cast<size_t>(height);
      OIDNContext* mutableThis = const_cast<OIDNContext*>(this);
      mutableThis->m_oidnPackedColorBuffer.resize(packedByteSize);

      pFilterData = mutableThis->m_oidnPackedColorBuffer.data();
      filterRowPitch = tightRowPitch;

      for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(
          pFilterData + filterRowPitch * static_cast<size_t>(y),
          pData + rowPitch * static_cast<size_t>(y),
          tightRowPitch);
      }
    }

    if (!ensureOidnFilterResources(width, height, filterRowPitch, format, pixelStride, nullptr, nullptr, 0, false)) {
      return false;
    }

    const size_t byteSize = filterRowPitch * static_cast<size_t>(height);

    oidn::dispatch.WriteBuffer(m_oidnColorBuffer, 0, byteSize, pFilterData);
    if (!checkDeviceError("uploading color buffer")) {
      return false;
    }

    oidn::dispatch.ExecuteFilter(m_oidnFilter);
    if (!checkDeviceError("executing filter")) {
      return false;
    }

    oidn::dispatch.ReadBuffer(m_oidnColorBuffer, 0, byteSize, pFilterData);
    if (!checkDeviceError("downloading output buffer")) {
      return false;
    }

    if (pFilterData != pData) {
      for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(
          pData + rowPitch * static_cast<size_t>(y),
          pFilterData + filterRowPitch * static_cast<size_t>(y),
          tightRowPitch);
      }
    }

    return true;
  }

  void OIDNContext::dispatch(
    Rc<DxvkContext> ctx,
    DxvkBarrierSet& barriers,
    const Resources::RaytracingOutput& rtOutput,
    const DxvkDenoise::Input& inputs,
    const DxvkDenoise::Output& outputs) {
    (void)barriers;

    if (!m_isAvailable || m_oidnDevice == nullptr || m_syncSignal == nullptr) {
      return;
    }

    if (inputs.reset) {
      resetTemporalHistory();
    }

    std::array<FilterTarget, 2> targets = {};
    uint32_t targetCount = 0;

    auto appendTarget = [&](const Resources::Resource* input, const Resources::Resource* output, uint32_t slot) {
      if (input == nullptr || output == nullptr || targetCount >= targets.size()) {
        return;
      }

      FilterTarget& target = targets[targetCount++];
      target.input = input;
      target.output = output;
      target.slot = slot;
    };

    const bool optimizeSecondariesForFast =
      m_type == DenoiserType::Secondaries
      && RtxOptions::oidnQuality() == OidnQuality::Fast;

    if (inputs.reference != nullptr && outputs.reference != nullptr) {
      appendTarget(inputs.reference, outputs.reference, 0);
    } else if (optimizeSecondariesForFast) {
      // Fast mode keeps a lower-cost secondaries path by denoising only specular lobe.
      if (inputs.diffuse_hitT != nullptr && outputs.diffuse_hitT != nullptr) {
        copyImage(ctx,
          outputs.diffuse_hitT->image,
          inputs.diffuse_hitT->image,
          inputs.diffuse_hitT->image->info().extent);
      }

      appendTarget(inputs.specular_hitT, outputs.specular_hitT, 1);
    } else {
      appendTarget(inputs.diffuse_hitT, outputs.diffuse_hitT, 0);
      appendTarget(inputs.specular_hitT, outputs.specular_hitT, 1);
    }

    if (targetCount == 0) {
      return;
    }

    bool hasSupportedTarget = false;

    for (uint32_t i = 0; i < targetCount; ++i) {
      FilterTarget& target = targets[i];

      target.supported = getSupportedOidnFormat(
        target.input->image->info().format,
        target.format,
        target.pixelStride);

      if (!target.supported) {
        if (!m_loggedUnsupportedFormat) {
          Logger::warn("[RTX] OIDN: Unsupported input format, falling back to unfiltered signal for this target");
          m_loggedUnsupportedFormat = true;
        }

        copyImage(ctx,
          target.output->image,
          target.input->image,
          target.input->image->info().extent);
        continue;
      }

      if (!ensureStagingImage(ctx, target.slot, *target.input)) {
        copyImage(ctx,
          target.output->image,
          target.input->image,
          target.input->image->info().extent);
        continue;
      }

      copyImage(ctx,
        m_stagingImages[target.slot].image,
        target.input->image,
        target.input->image->info().extent);

      hasSupportedTarget = true;
    }

    if (!hasSupportedTarget) {
      return;
    }

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT);

    const uint64_t syncValue = ++m_syncValue;
    ctx->signal(m_syncSignal, syncValue);
    ctx->flushCommandList();
    m_syncSignal->wait(syncValue);

    for (uint32_t i = 0; i < targetCount; ++i) {
      FilterTarget& target = targets[i];
      if (!target.supported) {
        continue;
      }

      uint8_t* pData = nullptr;
      const StagingImage& staging = m_stagingImages[target.slot];

      VkImageSubresource subresource = {};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.mipLevel = 0;
      subresource.arrayLayer = 0;

      const VkSubresourceLayout layout = staging.image->querySubresourceLayout(subresource);

      target.filtered = runOidnFilter(
        &staging,
        nullptr,
        layout.offset,
        pData,
        staging.extent.width,
        staging.extent.height,
        layout.rowPitch,
        target.format,
        target.pixelStride);

      if (!target.filtered) {
        if (target.slot < m_temporalHistory.size()) {
          m_temporalHistory[target.slot].valid = false;
        }

        Logger::warn("[RTX] OIDN: Filter execution failed, falling back to unfiltered signal for this target");
      }
    }

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_READ_BIT);

    for (uint32_t i = 0; i < targetCount; ++i) {
      const FilterTarget& target = targets[i];
      if (!target.supported) {
        continue;
      }

      if (target.filtered) {
        copyImage(ctx,
          target.output->image,
          m_stagingImages[target.slot].image,
          target.output->image->info().extent);
      } else {
        copyImage(ctx,
          target.output->image,
          target.input->image,
          target.input->image->info().extent);
      }
    }
  }

}  // namespace dxvk
