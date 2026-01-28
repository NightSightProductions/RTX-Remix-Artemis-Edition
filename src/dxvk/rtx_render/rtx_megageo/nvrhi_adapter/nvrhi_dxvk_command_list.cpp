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
// Enable verbose MegaGeo logging for debugging
#define RTXMG_VERBOSE_LOGGING 0
#if RTXMG_VERBOSE_LOGGING
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

#include "nvrhi_dxvk_command_list.h"
#include "nvrhi_dxvk_buffer.h"
#include "nvrhi_dxvk_texture.h"
#include "nvrhi_dxvk_pipeline.h"
#include "nvrhi_dxvk_sampler.h"
#include "../../../util/log/log.h"
#include <algorithm>

namespace dxvk {

  void NvrhiDxvkCommandList::open() {
    // DXVK contexts are always "open" - no-op
  }

  void NvrhiDxvkCommandList::close() {
    // Flush command list
    m_context->flushCommandList();
  }

  void NvrhiDxvkCommandList::clearState() {
    RTXMG_LOG("RTX MegaGeo: clearState - resetting compute state");
    m_computeState = nvrhi::ComputeState();
    // Reset scratch buffer write pointers at start of frame
    // DXVK ensures GPU is done with previous frame's work
    RTXMG_LOG("RTX MegaGeo: clearState - calling resetWritePointers");
    m_scratchManager->resetWritePointers();
    RTXMG_LOG("RTX MegaGeo: clearState - done");
  }

  void NvrhiDxvkCommandList::writeBuffer(
    nvrhi::IBuffer* buffer,
    const void* data,
    size_t size,
    uint64_t offset)
  {
    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

    // Cache data for small constant buffers (used for push constants)
    // Only cache if offset is 0 and size is <= 128 bytes
    if (offset == 0 && size <= 128) {
      nvrhiBuffer->setCachedData(data, size);
    }

    // Vulkan requires vkCmdUpdateBuffer size to be a multiple of 4
    // Round UP to multiple of 4, but only write within buffer bounds
    size_t bufferSize = nvrhiBuffer->getDesc().byteSize;
    size_t maxWriteSize = (offset < bufferSize) ? (bufferSize - offset) : 0;

    // Determine how much we can actually write
    size_t writeSize = std::min(size, maxWriteSize);
    // Align down to multiple of 4 (Vulkan requirement)
    writeSize = writeSize & ~3;

    if (writeSize == 0) {
      // Nothing to write or buffer too small
      if (size > 0) {
        Logger::warn(str::format("RTX MegaGeo: writeBuffer - cannot write ", size, " bytes at offset ", offset,
          " to buffer of size ", bufferSize));
      }
      return;
    }

    // Vulkan limits vkCmdUpdateBuffer to 65536 bytes - split large updates into chunks
    constexpr size_t maxChunkSize = 65536;

    if (writeSize <= maxChunkSize) {
      // Small update - send as single chunk
      m_context->updateBuffer(dxvkBuffer, offset, writeSize, data);
    } else {
      // Large update - split into chunks
      const uint8_t* dataPtr = static_cast<const uint8_t*>(data);
      size_t remaining = writeSize;
      uint64_t currentOffset = offset;

      while (remaining > 0) {
        size_t chunkSize = std::min(remaining, maxChunkSize);
        // Align chunk size down to 4 bytes for Vulkan
        chunkSize = chunkSize & ~3;
        if (chunkSize == 0) break;

        m_context->updateBuffer(dxvkBuffer, currentOffset, chunkSize, dataPtr);

        dataPtr += chunkSize;
        currentOffset += chunkSize;
        remaining -= chunkSize;
      }
    }
  }

  void NvrhiDxvkCommandList::clearBufferUInt(
    nvrhi::IBuffer* buffer,
    uint32_t value)
  {
    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

    // Clear buffer with uint32 value
    const nvrhi::BufferDesc& desc = nvrhiBuffer->getDesc();
    m_context->clearBuffer(dxvkBuffer, 0, desc.byteSize, value);
  }

  void NvrhiDxvkCommandList::copyBuffer(
    nvrhi::IBuffer* dst,
    uint64_t dstOffset,
    nvrhi::IBuffer* src,
    uint64_t srcOffset,
    uint64_t size)
  {
    NvrhiDxvkBuffer* nvrhiDst = static_cast<NvrhiDxvkBuffer*>(dst);
    NvrhiDxvkBuffer* nvrhiSrc = static_cast<NvrhiDxvkBuffer*>(src);

    Rc<DxvkBuffer> dxvkDst = nvrhiDst->getDxvkBuffer();
    Rc<DxvkBuffer> dxvkSrc = nvrhiSrc->getDxvkBuffer();

    m_context->copyBuffer(dxvkDst, dstOffset, dxvkSrc, srcOffset, size);
  }

  void NvrhiDxvkCommandList::copyTexture(nvrhi::ITexture* dst, nvrhi::ITexture* src) {
    NvrhiDxvkTexture* dstTex = static_cast<NvrhiDxvkTexture*>(dst);
    NvrhiDxvkTexture* srcTex = static_cast<NvrhiDxvkTexture*>(src);

    const auto& dstDesc = dstTex->getDesc();
    const auto& srcDesc = srcTex->getDesc();

    // Copy entire texture
    VkImageSubresourceLayers subresource;
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel = 0;
    subresource.baseArrayLayer = 0;
    subresource.layerCount = 1;

    VkExtent3D extent = { srcDesc.width, srcDesc.height, srcDesc.depth };

    m_context->copyImage(
      dstTex->getDxvkImage(), subresource, VkOffset3D{ 0, 0, 0 },
      srcTex->getDxvkImage(), subresource, VkOffset3D{ 0, 0, 0 },
      extent);
  }

  void NvrhiDxvkCommandList::clearTextureFloat(
    nvrhi::ITexture* texture,
    const nvrhi::TextureSubresourceSet& subresources,
    const nvrhi::Color& clearColor)
  {
    NvrhiDxvkTexture* tex = static_cast<NvrhiDxvkTexture*>(texture);

    VkClearColorValue vkClearColor;
    vkClearColor.float32[0] = clearColor.r;
    vkClearColor.float32[1] = clearColor.g;
    vkClearColor.float32[2] = clearColor.b;
    vkClearColor.float32[3] = clearColor.a;

    VkImageSubresourceRange range;
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = subresources.baseMipLevel;
    range.levelCount = (subresources.numMipLevels == nvrhi::TextureSubresourceSet::AllMipLevels) ?
                       VK_REMAINING_MIP_LEVELS : subresources.numMipLevels;
    range.baseArrayLayer = subresources.baseArraySlice;
    range.layerCount = (subresources.numArraySlices == nvrhi::TextureSubresourceSet::AllArraySlices) ?
                       VK_REMAINING_ARRAY_LAYERS : subresources.numArraySlices;

    m_context->clearColorImage(
      tex->getDxvkImage(),
      vkClearColor,
      range);
  }

  void NvrhiDxvkCommandList::setComputeState(const nvrhi::ComputeState& state) {
    m_computeState = state;
    bindComputeResources(state);
  }

  void NvrhiDxvkCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    // Ensure all prior writes are visible to this shader read
    m_context->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    m_context->dispatch(x, y, z);
  }

  void NvrhiDxvkCommandList::dispatchIndirect(nvrhi::IBuffer* buffer, uint64_t offset) {
    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

    // Bind indirect argument buffer
    DxvkBufferSlice argBufferSlice(dxvkBuffer, offset, sizeof(VkDispatchIndirectCommand));
    m_context->bindDrawBuffers(argBufferSlice, DxvkBufferSlice());
    m_context->dispatchIndirect(offset);
  }

  void NvrhiDxvkCommandList::dispatchIndirect(uint64_t offset) {
    // Use the indirect params buffer from the current compute state
    if (!m_computeState.indirectParamsBuffer) {
      Logger::err("dispatchIndirect(offset) called without setting indirect params buffer in compute state");
      return;
    }
    dispatchIndirect(m_computeState.indirectParamsBuffer.Get(), offset);
  }

  void NvrhiDxvkCommandList::executeMultiIndirectClusterOperation(
    const nvrhi::rt::cluster::OperationDesc& nvrhiDesc)
  {
    RTXMG_LOG("RTX MegaGeo: executeMultiIndirectClusterOperation - Entry");

    const nvrhi::rt::cluster::OperationDesc& desc = nvrhiDesc;

    // CRITICAL: Insert barriers matching NVRHI sample behavior
    // Sample uses: ShaderResource (SHADER_READ) for input, UnorderedAccess (SHADER_READ|SHADER_WRITE) for output
    if (desc.inIndirectArgsBuffer) {
      // indirectArgsBuffer: ShaderResource state (readable)
      m_context->emitMemoryBarrier(
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }
    if (desc.outSizesBuffer) {
      // outSizesBuffer: UnorderedAccess state (read/write)
      m_context->emitMemoryBarrier(
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    RTXMG_LOG("RTX MegaGeo: Inserted pre-command barriers");

    // Get direct Vulkan command buffer from DXVK
    VkCommandBuffer cmdBuffer = m_context->getCmdBuffer(DxvkCmdBuffer::ExecBuffer);
    RTXMG_LOG(str::format("RTX MegaGeo: Got command buffer: ", (void*)cmdBuffer));

    // Translate NVRHI descriptor to Vulkan cluster commands
    VkClusterAccelerationStructureCommandsInfoNV vkCmds = {};
    vkCmds.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    RTXMG_LOG("RTX MegaGeo: About to translate cluster operation");
    translateClusterOperation(desc, vkCmds);
    RTXMG_LOG("RTX MegaGeo: Cluster operation translated");

    // Load cluster AS extension function pointer manually
    // Note: This function is not in DXVK's InstanceFn table yet, so we load it dynamically
    typedef void (VKAPI_PTR *PFN_vkCmdBuildClusterAccelerationStructureIndirectNV)(
      VkCommandBuffer commandBuffer,
      const VkClusterAccelerationStructureCommandsInfoNV* pInfo);

    static PFN_vkCmdBuildClusterAccelerationStructureIndirectNV vkCmdBuildClusterAS = nullptr;
    if (!vkCmdBuildClusterAS) {
      RTXMG_LOG("RTX MegaGeo: Loading cluster AS extension function");
      // Load via device loader
      vkCmdBuildClusterAS = reinterpret_cast<PFN_vkCmdBuildClusterAccelerationStructureIndirectNV>(
        m_device->getDxvkDevice()->vkd()->sym("vkCmdBuildClusterAccelerationStructureIndirectNV"));

      if (vkCmdBuildClusterAS) {
        RTXMG_LOG(str::format("RTX MegaGeo: Cluster AS function loaded at: ", (void*)vkCmdBuildClusterAS));
      }
    }

    if (vkCmdBuildClusterAS) {
      RTXMG_LOG("RTX MegaGeo: About to call vkCmdBuildClusterAccelerationStructureIndirectNV");
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.sType=", vkCmds.sType, " vkCmds.pNext=", (void*)vkCmds.pNext));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.scratchData=", vkCmds.scratchData,
                               " vkCmds.dstImplicitData=", vkCmds.dstImplicitData,
                               " vkCmds.srcInfosCount=", vkCmds.srcInfosCount,
                               " vkCmds.addressResolutionFlags=", vkCmds.addressResolutionFlags));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.srcInfosArray.deviceAddress=", vkCmds.srcInfosArray.deviceAddress,
                               " stride=", vkCmds.srcInfosArray.stride,
                               " size=", vkCmds.srcInfosArray.size));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.dstAddressesArray.deviceAddress=", vkCmds.dstAddressesArray.deviceAddress,
                               " stride=", vkCmds.dstAddressesArray.stride,
                               " size=", vkCmds.dstAddressesArray.size));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.dstSizesArray.deviceAddress=", vkCmds.dstSizesArray.deviceAddress,
                               " stride=", vkCmds.dstSizesArray.stride,
                               " size=", vkCmds.dstSizesArray.size));
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.input.sType=", vkCmds.input.sType,
                               " maxAccelStructCount=", vkCmds.input.maxAccelerationStructureCount,
                               " opType=", (uint32_t)vkCmds.input.opType,
                               " opMode=", (uint32_t)vkCmds.input.opMode));
      // CRITICAL: Log the opInput union pointers to verify they're set
      RTXMG_LOG(str::format("RTX MegaGeo: FINAL vkCmds.input.opInput.pTriangleClusters=",
                               (void*)vkCmds.input.opInput.pTriangleClusters,
                               " pClustersBottomLevel=", (void*)vkCmds.input.opInput.pClustersBottomLevel,
                               " pMoveObjects=", (void*)vkCmds.input.opInput.pMoveObjects));
      RTXMG_LOG("RTX MegaGeo: Calling vkCmdBuildClusterAccelerationStructureIndirectNV");
      vkCmdBuildClusterAS(cmdBuffer, &vkCmds);
      RTXMG_LOG("RTX MegaGeo: vkCmdBuildClusterAccelerationStructureIndirectNV completed");
    } else {
      Logger::err("VK_NV_cluster_acceleration_structure extension function not available");
      return;
    }

    // Insert post-operation barrier to ensure cluster AS build completes before any dependent operations
    // This is critical for synchronizing between the cluster AS build and subsequent TLAS builds or ray tracing
    m_context->emitMemoryBarrier(
      0,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT);
    RTXMG_LOG("RTX MegaGeo: Inserted post-operation barrier");

    RTXMG_LOG("RTX MegaGeo: executeMultiIndirectClusterOperation - Complete");
  }

  void NvrhiDxvkCommandList::bufferBarrier(
    nvrhi::IBuffer* buffer,
    nvrhi::ResourceStates stateBefore,
    nvrhi::ResourceStates stateAfter)
  {
    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

    // Translate NVRHI resource states to Vulkan stages/access
    DxvkAccessFlags srcAccess(0);
    DxvkAccessFlags dstAccess(0);
    VkPipelineStageFlags srcStages = 0;
    VkPipelineStageFlags dstStages = 0;

    // Source state
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      srcAccess.set(DxvkAccess::Read);
      srcAccess.set(DxvkAccess::Write);
      srcStages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::IndirectArgument) != 0) {
      srcAccess.set(DxvkAccess::Read);
      srcStages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    }
    if (static_cast<uint32_t>(stateBefore & nvrhi::ResourceStates::AccelStructBuildInput) != 0) {
      srcAccess.set(DxvkAccess::Read);
      srcStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }

    // Destination state
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::UnorderedAccess) != 0) {
      dstAccess.set(DxvkAccess::Read);
      dstAccess.set(DxvkAccess::Write);
      dstStages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::IndirectArgument) != 0) {
      dstAccess.set(DxvkAccess::Read);
      dstStages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    }
    if (static_cast<uint32_t>(stateAfter & nvrhi::ResourceStates::AccelStructBuildInput) != 0) {
      dstAccess.set(DxvkAccess::Read);
      dstStages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }

    // Emit memory barrier (DXVK doesn't have emitBufferBarrier, use memory barrier instead)
    m_context->emitMemoryBarrier(0, srcStages, srcAccess.raw(), dstStages, dstAccess.raw());
  }

  void NvrhiDxvkCommandList::globalBarrier(
    nvrhi::ResourceStates stateBefore,
    nvrhi::ResourceStates stateAfter)
  {
    // Similar translation as bufferBarrier, but for global memory barrier
    VkPipelineStageFlags srcStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dstStages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkAccessFlags srcAccess = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkAccessFlags dstAccess = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    m_context->emitMemoryBarrier(0, srcStages, srcAccess, dstStages, dstAccess);
  }

  void NvrhiDxvkCommandList::bindComputeResources(const nvrhi::ComputeState& state) {
    // Bind the compute shader from the pipeline
    if (state.pipeline) {
      auto* computePipeline = static_cast<NvrhiDxvkComputePipeline*>(state.pipeline.Get());
      const Rc<DxvkShader>& shader = computePipeline->getShader();
      if (shader != nullptr) {
        m_context->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, shader);
        RTXMG_LOG(str::format("RTX MegaGeo: bindComputeResources - bound shader"));
      }
    }

    // Bind resources from binding sets
    // DXVK slot mapping (from Slang/HLSL register to Vulkan binding):
    //   t0-t99 (SRVs)     -> slots 0-99
    //   s0-s99 (Samplers) -> slots 100-199
    //   u0-u99 (UAVs)     -> slots 200-299
    //   b0-b99 (CBVs)     -> slots 300-399
    uint32_t bindingSetIndex = 0;
    for (const auto& bindingSet : state.bindingSets) {
      if (!bindingSet) {
        RTXMG_LOG(str::format("RTX MegaGeo: bindComputeResources - bindingSet[", bindingSetIndex, "] is null"));
        bindingSetIndex++;
        continue;
      }

      auto* nvrhiBindingSet = static_cast<NvrhiDxvkBindingSet*>(bindingSet.Get());
      const nvrhi::BindingSetDesc& desc = nvrhiBindingSet->getDesc();
      RTXMG_LOG(str::format("RTX MegaGeo: bindComputeResources - bindingSet[", bindingSetIndex, "] has ", desc.bindings.size(), " bindings"));

      for (const auto& item : desc.bindings) {
        uint32_t dxvkSlot = item.slot;  // Start with HLSL register number

        switch (item.type) {
          case nvrhi::BindingSetItem::Type::StructuredBuffer_SRV:
            // SRVs: t register -> slot 0-99
            // dxvkSlot = item.slot (already correct)
            if (item.resourceHandle) {
              auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
              Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
              uint64_t bufferSize = dxvkBuffer->info().size;
              // Resolve range to fit buffer (matching NVRHI's BufferRange::resolve)
              uint64_t offset = std::min(item.range.byteOffset, bufferSize);
              uint64_t size = (item.range.byteSize > 0)
                ? std::min(item.range.byteSize, bufferSize - offset)
                : bufferSize - offset;
              DxvkBufferSlice slice(dxvkBuffer, offset, size);
              // Get the actual slice handle to see physical offset
              DxvkBufferSliceHandle sliceHandle = slice.getSliceHandle();
              uint64_t physOffset = sliceHandle.offset;
              if (physOffset % 16 != 0) {
                Logger::err(str::format("RTX MegaGeo: ALIGNMENT ERROR! SRV slot=", dxvkSlot,
                  " physOffset=", physOffset, " (not 16-byte aligned!)",
                  " rangeOffset=", offset, " buffer=", nvrhiBuffer->getDesc().debugName));
              }
              RTXMG_LOG(str::format("RTX MegaGeo: bind SRV slot=", dxvkSlot, " physOffset=", physOffset, " rangeOffset=", offset, " size=", size));
              m_context->bindResourceBuffer(dxvkSlot, slice);
            } else {
              Logger::err(str::format("RTX MegaGeo: SRV slot=", dxvkSlot, " has NULL resourceHandle!"));
            }
            break;

          case nvrhi::BindingSetItem::Type::StructuredBuffer_UAV:
            // UAVs: u register -> slot 200-299
            dxvkSlot = 200 + item.slot;
            if (item.resourceHandle) {
              auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);
              Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
              uint64_t bufferSize = dxvkBuffer->info().size;
              // Resolve range to fit buffer (matching NVRHI's BufferRange::resolve)
              uint64_t offset = std::min(item.range.byteOffset, bufferSize);
              uint64_t size = (item.range.byteSize > 0)
                ? std::min(item.range.byteSize, bufferSize - offset)
                : bufferSize - offset;
              DxvkBufferSlice slice(dxvkBuffer, offset, size);
              // Get the actual slice handle to see physical offset
              DxvkBufferSliceHandle sliceHandle = slice.getSliceHandle();
              uint64_t physOffset = sliceHandle.offset;
              if (physOffset % 16 != 0) {
                Logger::err(str::format("RTX MegaGeo: ALIGNMENT ERROR! UAV slot=", dxvkSlot,
                  " physOffset=", physOffset, " (not 16-byte aligned!)",
                  " rangeOffset=", offset, " buffer=", nvrhiBuffer->getDesc().debugName));
              }
              RTXMG_LOG(str::format("RTX MegaGeo: bind UAV slot=", dxvkSlot, " (u", item.slot, ") physOffset=", physOffset, " rangeOffset=", offset, " size=", size));
              m_context->bindResourceBuffer(dxvkSlot, slice);
            }
            break;

          case nvrhi::BindingSetItem::Type::ConstantBuffer:
            // Constant buffers: Use push constants if data is cached, otherwise bind as uniform buffer
            dxvkSlot = 300 + item.slot;
            if (item.resourceHandle) {
              auto* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(item.resourceHandle);

              // CRITICAL: If we have cached data, push it as push constants
              // This is needed because Slang may compile ConstantBuffer to push constants in SPIR-V
              if (nvrhiBuffer->hasCachedData()) {
                uint32_t pushOffset = 0;  // Push constants always start at offset 0
                uint32_t pushSize = static_cast<uint32_t>(nvrhiBuffer->getCachedDataSize());
                RTXMG_LOG(str::format("RTX MegaGeo: pushConstants size=", pushSize, " (b", item.slot, ")"));
                m_context->pushConstants(pushOffset, pushSize, nvrhiBuffer->getCachedData());
              }

              // Also bind as uniform buffer (in case shader uses descriptor binding)
              Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();
              uint64_t bufferSize = dxvkBuffer->info().size;
              // Resolve range to fit buffer (matching NVRHI's BufferRange::resolve)
              uint64_t offset = std::min(item.range.byteOffset, bufferSize);
              uint64_t size = (item.range.byteSize > 0)
                ? std::min(item.range.byteSize, bufferSize - offset)
                : bufferSize - offset;
              DxvkBufferSlice slice(dxvkBuffer, offset, size);
              RTXMG_LOG(str::format("RTX MegaGeo: bind CBV slot=", dxvkSlot, " (b", item.slot, ") offset=", offset, " size=", size));
              m_context->bindResourceBuffer(dxvkSlot, slice);
            }
            break;

          case nvrhi::BindingSetItem::Type::Sampler:
            // Samplers: s register -> slot 100-199
            dxvkSlot = 100 + item.slot;
            if (item.resourceHandle) {
              auto* nvrhiSampler = static_cast<NvrhiDxvkSampler*>(item.resourceHandle);
              Rc<DxvkSampler> dxvkSampler = nvrhiSampler->getDxvkSampler();
              RTXMG_LOG(str::format("RTX MegaGeo: bind Sampler slot=", dxvkSlot, " (s", item.slot, ")"));
              m_context->bindResourceSampler(dxvkSlot, dxvkSampler);
            }
            break;

          case nvrhi::BindingSetItem::Type::Texture_SRV:
            // Texture SRVs: t register -> slot 0-99 (same as buffer SRVs)
            // For array elements, add the array index to the base slot
            {
              uint32_t effectiveSlot = dxvkSlot + item.arrayElement;
              if (item.resourceHandle) {
                auto* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(item.resourceHandle);
                const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();

                // Create image view for shader resource
                DxvkImageViewCreateInfo viewInfo;
                viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = dxvkImage->info().format;
                viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
                viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                viewInfo.minLevel = 0;
                viewInfo.numLevels = dxvkImage->info().mipLevels;
                viewInfo.minLayer = 0;
                viewInfo.numLayers = 1;

                Rc<DxvkImageView> imageView = m_device->getDxvkDevice()->createImageView(dxvkImage, viewInfo);
                RTXMG_LOG(str::format("RTX MegaGeo: bind Texture_SRV slot=", effectiveSlot, " (t", item.slot, " array=", item.arrayElement, ")"));
                m_context->bindResourceView(effectiveSlot, imageView, nullptr);
              } else {
                Logger::err(str::format("RTX MegaGeo: Texture_SRV slot=", effectiveSlot, " has NULL resourceHandle!"));
              }
            }
            break;

          default:
            Logger::warn(str::format("bindComputeResources: Unhandled binding type ", (int)item.type));
            break;
        }
      }
      bindingSetIndex++;
    }
  }

  void NvrhiDxvkCommandList::translateClusterOperation(
    const nvrhi::rt::cluster::OperationDesc& desc,
    VkClusterAccelerationStructureCommandsInfoNV& vkCmds)
  {
    // NOTE: The Vulkan cluster AS API has changed significantly from the original SDK version.
    // The actual implementation needs adaptation based on RTX Remix's usage patterns.

    memset(&vkCmds, 0, sizeof(vkCmds));
    vkCmds.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV;
    vkCmds.pNext = nullptr;  // Explicitly set pNext

    // Set up operation input info
    VkClusterAccelerationStructureInputInfoNV& input = vkCmds.input;
    memset(&input, 0, sizeof(input));  // Zero-initialize input structure
    input.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    input.pNext = nullptr;  // Explicitly set pNext
    input.maxAccelerationStructureCount = desc.params.maxArgCount;

    // Map operation type (NOTE: enum values have changed in the new API)
    const char* opTypeName = "Unknown";
    switch (desc.params.type) {
      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
        input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        opTypeName = "ClasBuildTemplates";
        break;
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
        opTypeName = "ClasInstantiateTemplates";
        break;
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        input.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
        opTypeName = "BlasBuild";
        break;
    }

    // Map operation mode (NOTE: enum values have changed in the new API)
    const char* opModeName = "Unknown";
    switch (desc.params.mode) {
      case nvrhi::rt::cluster::OperationMode::GetSizes:
        input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
        opModeName = "GetSizes";
        break;
      case nvrhi::rt::cluster::OperationMode::ExplicitDestinations:
        input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        opModeName = "ExplicitDestinations";
        break;
      case nvrhi::rt::cluster::OperationMode::ImplicitDestinations:
        input.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
        opModeName = "ImplicitDestinations";
        break;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Operation type=", opTypeName, " mode=", opModeName,
                             " maxArgCount=", input.maxAccelerationStructureCount));
    RTXMG_LOG(str::format("RTX MegaGeo: input.sType=", input.sType, " input.pNext=", (void*)input.pNext,
                             " input.opType=", (uint32_t)input.opType, " input.opMode=", (uint32_t)input.opMode));

    // Set up opInput based on operation type
    // Note: These structures need to persist for the lifetime of the command
    static thread_local VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput;
    static thread_local VkClusterAccelerationStructureTriangleClusterInputNV clusterInput;

    // CRITICAL: Zero the entire opInput union before setting any member
    memset(&input.opInput, 0, sizeof(input.opInput));

    switch (desc.params.type) {
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        // CRITICAL: Explicitly zero and initialize to clear any stale data
        memset(&blasInput, 0, sizeof(blasInput));
        blasInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
        blasInput.pNext = nullptr;
        blasInput.maxTotalClusterCount = desc.params.blas.maxTotalClasCount;
        blasInput.maxClusterCountPerAccelerationStructure = desc.params.blas.maxClasPerBlasCount;
        input.opInput.pClustersBottomLevel = &blasInput;
        RTXMG_LOG(str::format("RTX MegaGeo: BlasBuild maxTotalClusterCount=", blasInput.maxTotalClusterCount,
                                 " maxClusterCountPerAS=", blasInput.maxClusterCountPerAccelerationStructure));
        RTXMG_LOG(str::format("RTX MegaGeo: blasInput.sType=", blasInput.sType, " blasInput.pNext=", (void*)blasInput.pNext,
                                 " opInput.pClustersBottomLevel=", (void*)input.opInput.pClustersBottomLevel));
        break;

      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        // CRITICAL: Explicitly zero and initialize to clear any stale data
        memset(&clusterInput, 0, sizeof(clusterInput));
        clusterInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
        clusterInput.pNext = nullptr;
        clusterInput.vertexFormat = desc.params.clas.vertexFormat;
        clusterInput.maxGeometryIndexValue = desc.params.clas.maxGeometryIndex;
        clusterInput.maxClusterUniqueGeometryCount = desc.params.clas.maxUniqueGeometryCount;
        clusterInput.maxClusterTriangleCount = desc.params.clas.maxTriangleCount;
        clusterInput.maxClusterVertexCount = desc.params.clas.maxVertexCount;
        clusterInput.maxTotalTriangleCount = desc.params.clas.maxTotalTriangleCount;
        clusterInput.maxTotalVertexCount = desc.params.clas.maxTotalVertexCount;
        clusterInput.minPositionTruncateBitCount = desc.params.clas.minPositionTruncateBitCount;
        input.opInput.pTriangleClusters = &clusterInput;
        RTXMG_LOG(str::format("RTX MegaGeo: ClusterOp maxTriCount=", clusterInput.maxTotalTriangleCount,
                                 " maxVertCount=", clusterInput.maxTotalVertexCount));
        RTXMG_LOG(str::format("RTX MegaGeo: clusterInput.sType=", clusterInput.sType, " clusterInput.pNext=", (void*)clusterInput.pNext,
                                 " vertexFormat=", (uint32_t)clusterInput.vertexFormat,
                                 " maxGeometryIndexValue=", clusterInput.maxGeometryIndexValue,
                                 " maxClusterUniqueGeometryCount=", clusterInput.maxClusterUniqueGeometryCount,
                                 " opInput.pTriangleClusters=", (void*)input.opInput.pTriangleClusters));
        RTXMG_LOG(str::format("RTX MegaGeo: clusterInput sizeof=", sizeof(clusterInput),
                                 " maxClusterTriangleCount=", clusterInput.maxClusterTriangleCount,
                                 " maxClusterVertexCount=", clusterInput.maxClusterVertexCount,
                                 " minPositionTruncateBitCount=", clusterInput.minPositionTruncateBitCount));
        break;
    }

    // Set up srcInfosArray (indirect args buffer)
    if (desc.inIndirectArgsBuffer) {
      NvrhiDxvkBuffer* argsBuffer = static_cast<NvrhiDxvkBuffer*>(desc.inIndirectArgsBuffer);
      vkCmds.srcInfosArray.deviceAddress = argsBuffer->getGpuVirtualAddress() + desc.inIndirectArgsOffsetInBytes;
      vkCmds.srcInfosArray.stride = argsBuffer->getDesc().structStride;
      vkCmds.srcInfosArray.size = argsBuffer->getDesc().byteSize - desc.inIndirectArgsOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: srcInfosArray addr=", vkCmds.srcInfosArray.deviceAddress,
                               " stride=", vkCmds.srcInfosArray.stride, " size=", vkCmds.srcInfosArray.size));
    } else {
      Logger::warn("RTX MegaGeo: inIndirectArgsBuffer is NULL");
    }

    // Set up inOutAddressesBuffer (BLAS/CLAS addresses)
    // For GetSizes mode, this should be NULL/zero
    if (desc.inOutAddressesBuffer) {
      NvrhiDxvkBuffer* addressesBuffer = static_cast<NvrhiDxvkBuffer*>(desc.inOutAddressesBuffer);
      vkCmds.dstAddressesArray.deviceAddress = addressesBuffer->getGpuVirtualAddress() + desc.inOutAddressesOffsetInBytes;
      // Use buffer's structStride instead of hardcoding sizeof(VkDeviceAddress)
      vkCmds.dstAddressesArray.stride = addressesBuffer->getDesc().structStride;
      vkCmds.dstAddressesArray.size = addressesBuffer->getDesc().byteSize - desc.inOutAddressesOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: dstAddressesArray addr=", vkCmds.dstAddressesArray.deviceAddress,
                               " stride=", vkCmds.dstAddressesArray.stride, " size=", vkCmds.dstAddressesArray.size));
    } else {
      // Explicitly zero the array for GetSizes mode
      vkCmds.dstAddressesArray.deviceAddress = 0;
      vkCmds.dstAddressesArray.stride = 0;
      vkCmds.dstAddressesArray.size = 0;
      RTXMG_LOG("RTX MegaGeo: inOutAddressesBuffer is NULL (expected for GetSizes mode)");
    }

    // Set up output acceleration structures buffer
    // For GetSizes mode, this should be NULL/zero
    if (desc.outAccelerationStructuresBuffer) {
      NvrhiDxvkBuffer* accelBuffer = static_cast<NvrhiDxvkBuffer*>(desc.outAccelerationStructuresBuffer);
      vkCmds.dstImplicitData = accelBuffer->getGpuVirtualAddress() + desc.outAccelerationStructuresOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: dstImplicitData addr=", vkCmds.dstImplicitData));
    } else {
      // Explicitly zero for GetSizes mode
      vkCmds.dstImplicitData = 0;
      RTXMG_LOG("RTX MegaGeo: outAccelerationStructuresBuffer is NULL (expected for GetSizes mode)");
    }

    // Set up output sizes buffer
    // For GetSizes operations (outAccelerationStructuresBuffer is NULL), outSizesBuffer should be non-NULL
    // For Build operations (outAccelerationStructuresBuffer is non-NULL), outSizesBuffer can be NULL
    if (desc.outSizesBuffer) {
      NvrhiDxvkBuffer* sizesBuffer = static_cast<NvrhiDxvkBuffer*>(desc.outSizesBuffer);
      vkCmds.dstSizesArray.deviceAddress = sizesBuffer->getGpuVirtualAddress() + desc.outSizesOffsetInBytes;
      // Use buffer's structStride instead of hardcoding sizeof(uint32_t)
      vkCmds.dstSizesArray.stride = sizesBuffer->getDesc().structStride;
      vkCmds.dstSizesArray.size = sizesBuffer->getDesc().byteSize - desc.outSizesOffsetInBytes;
      RTXMG_LOG(str::format("RTX MegaGeo: dstSizesArray addr=", vkCmds.dstSizesArray.deviceAddress,
                               " stride=", vkCmds.dstSizesArray.stride, " size=", vkCmds.dstSizesArray.size,
                               " (numElements=", vkCmds.dstSizesArray.size / vkCmds.dstSizesArray.stride, ")"));
    } else if (!desc.outAccelerationStructuresBuffer && !desc.inOutAddressesBuffer) {
      // GetSizes mode but no sizes output - this is unexpected
      Logger::warn("RTX MegaGeo: outSizesBuffer is NULL in GetSizes mode");
    } else {
      // Build mode - outSizesBuffer being NULL is expected
      RTXMG_LOG("RTX MegaGeo: outSizesBuffer is NULL (expected for Build mode)");
    }

    // Set up scratch data
    if (desc.outScratchBuffer) {
      NvrhiDxvkBuffer* scratchBuffer = static_cast<NvrhiDxvkBuffer*>(desc.outScratchBuffer);
      vkCmds.scratchData = scratchBuffer->getGpuVirtualAddress();
      RTXMG_LOG(str::format("RTX MegaGeo: scratchData addr=", vkCmds.scratchData,
                               " size=", desc.scratchSizeInBytes));
    } else if (desc.scratchSizeInBytes > 0) {
      // Suballocate scratch buffer from pool (matching NVRHI's scratch manager behavior)
      Rc<DxvkBuffer> scratchBuffer;
      uint64_t scratchOffset = 0;
      constexpr uint32_t kClusterScratchAlignment = 256;

      if (!m_scratchManager->suballocateBuffer(desc.scratchSizeInBytes, &scratchBuffer, &scratchOffset,
                                                kClusterScratchAlignment)) {
        Logger::err(str::format("RTX MegaGeo: Failed to suballocate scratch buffer, size=", desc.scratchSizeInBytes));
        return;
      }

      vkCmds.scratchData = scratchBuffer->getDeviceAddress() + scratchOffset;
      RTXMG_LOG(str::format("RTX MegaGeo: suballocated scratch buffer, addr=", std::hex, vkCmds.scratchData,
                               " offset=", std::dec, scratchOffset,
                               " (requested=", desc.scratchSizeInBytes, ")"));

      // Track the buffer for command list lifetime
      m_context->getCommandList()->trackResource<DxvkAccess::Write>(scratchBuffer);
    } else {
      vkCmds.scratchData = 0;
      RTXMG_LOG("RTX MegaGeo: scratchSizeInBytes is 0, no scratch needed");
    }

    // Set srcInfosCount if indirect arg count buffer is provided
    if (desc.inIndirectArgCountBuffer) {
      NvrhiDxvkBuffer* countBuffer = static_cast<NvrhiDxvkBuffer*>(desc.inIndirectArgCountBuffer);
      vkCmds.srcInfosCount = countBuffer->getGpuVirtualAddress() + desc.inIndirectArgCountOffsetInBytes;
    } else {
      vkCmds.srcInfosCount = 0;
    }

    // Set address resolution flags to none for now
    vkCmds.addressResolutionFlags = 0;
  }

} // namespace dxvk
