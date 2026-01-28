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
// Disable verbose MegaGeo logging
#define RTXMG_VERBOSE_LOGGING 0
#if RTXMG_VERBOSE_LOGGING
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

#include "nvrhi_dxvk_device.h"
#include "nvrhi_dxvk_command_list.h"
#include "nvrhi_dxvk_sampler.h"
#include "nvrhi_dxvk_texture.h"
#include "nvrhi_dxvk_pipeline.h"
#include "nvrhi_dxvk_shader.h"
#include "../../../util/log/log.h"

namespace dxvk {

  nvrhi::BufferHandle NvrhiDxvkDevice::createBuffer(const nvrhi::BufferDesc& desc) {
    // Translate NVRHI buffer descriptor to DXVK buffer create info
    DxvkBufferCreateInfo dxvkInfo;
    dxvkInfo.size = desc.byteSize;
    dxvkInfo.usage = 0;
    dxvkInfo.stages = 0;
    dxvkInfo.access = 0;

    // Translate usage flags
    if (desc.isConstantBuffer) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dxvkInfo.access |= VK_ACCESS_UNIFORM_READ_BIT;
    }

    if (desc.canHaveUAVs || desc.canHaveRawViews || desc.canHaveTypedViews) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      dxvkInfo.access |= VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }

    if (desc.isVertexBuffer) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }

    if (desc.isIndexBuffer) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }

    if (desc.isDrawIndirectArgs) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      dxvkInfo.access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }

    if (desc.isAccelStructBuildInput) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      dxvkInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      dxvkInfo.access |= VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    }

    if (desc.isAccelStructStorage) {
      dxvkInfo.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
      dxvkInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      dxvkInfo.access |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    }

    // All buffers used in RTX MG need device address for cluster operations
    dxvkInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    dxvkInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // Default stages if none set
    if (dxvkInfo.stages == 0) {
      dxvkInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    }

    // Default access if none set
    if (dxvkInfo.access == 0) {
      dxvkInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }

    // Create DXVK buffer (GPU local memory by default)
    // Note: Debug name is set during buffer creation, not after
    const char* debugName = desc.debugName ? desc.debugName : "NVRHI Buffer";

    // DEBUG: Log buffer creation details
    RTXMG_LOG(str::format("RTX MegaGeo: Creating buffer '", debugName, "' size=", desc.byteSize,
                             " usage=0x", std::hex, dxvkInfo.usage, std::dec));

    // Determine memory type based on buffer purpose
    VkMemoryPropertyFlags memoryFlags;

    // Readback/upload buffers need HOST_VISIBLE memory so CPU can map and read/write them
    // Detect by checking cpuAccess mode or CopyDest initial state
    if (desc.cpuAccess == nvrhi::CpuAccessMode::Read ||
        desc.cpuAccess == nvrhi::CpuAccessMode::Write ||
        desc.cpuAccess == nvrhi::CpuAccessMode::ReadWrite ||
        desc.initialState == nvrhi::ResourceStates::CopyDest) {
      memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    } else {
      // CRITICAL: For cluster template buffers that need device addresses for GPU ray tracing,
      // we MUST allocate from pure device-local memory, not HVV (host-visible VRAM).
      // Using HVV causes low GPU addresses (~218MB) which leads to crashes.
      memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    Rc<DxvkBuffer> dxvkBuffer = m_device->createBuffer(
      dxvkInfo,
      memoryFlags,
      DxvkMemoryStats::Category::RTXAccelerationStructure,
      debugName);

    if (dxvkBuffer == nullptr) {
      Logger::err(str::format("Failed to create buffer of size ", desc.byteSize));
      return nullptr;
    }

    // DEBUG: Log device address and slice offset after creation
    VkDeviceAddress addr = dxvkBuffer->getDeviceAddress();
    DxvkBufferSliceHandle sliceHandle = dxvkBuffer->getSliceHandle();
    RTXMG_LOG(str::format("RTX MegaGeo: Created buffer '", debugName, "' with device address: ", addr));

    // Check alignment of the physical slice offset
    if (sliceHandle.offset % 16 != 0) {
      Logger::err(str::format("RTX MegaGeo: WARNING - Buffer '", debugName,
        "' created with non-16-byte-aligned slice offset: ", sliceHandle.offset,
        " (mod 16 = ", sliceHandle.offset % 16, ")"));
    }

    return new NvrhiDxvkBuffer(desc, dxvkBuffer);
  }

  void* NvrhiDxvkDevice::mapBuffer(nvrhi::IBuffer* buffer, nvrhi::CpuAccessMode access) {
    NvrhiDxvkBuffer* nvrhiBuffer = static_cast<NvrhiDxvkBuffer*>(buffer);
    Rc<DxvkBuffer> dxvkBuffer = nvrhiBuffer->getDxvkBuffer();

    // Map the entire buffer
    DxvkBufferSliceHandle slice = dxvkBuffer->getSliceHandle();
    return dxvkBuffer->mapPtr(0);
  }

  void NvrhiDxvkDevice::unmapBuffer(nvrhi::IBuffer* buffer) {
    // DXVK handles unmapping automatically via DxvkStagingDataAlloc
    // No explicit unmap needed
  }

  nvrhi::TextureHandle NvrhiDxvkDevice::createTexture(const nvrhi::TextureDesc& desc) {
    // Translate NVRHI texture descriptor to DXVK image create info
    DxvkImageCreateInfo dxvkInfo;

    dxvkInfo.type = VK_IMAGE_TYPE_2D;
    dxvkInfo.format = static_cast<VkFormat>(desc.format);
    dxvkInfo.flags = 0;
    dxvkInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    dxvkInfo.extent = { desc.width, desc.height, desc.depth };
    dxvkInfo.numLayers = desc.arraySize;
    dxvkInfo.mipLevels = desc.mipLevels;
    dxvkInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dxvkInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dxvkInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    dxvkInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

    if (desc.isUAV) {
      dxvkInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      dxvkInfo.access |= VK_ACCESS_SHADER_WRITE_BIT;
      // UAV images need GENERAL layout for compute storage access
      dxvkInfo.layout = VK_IMAGE_LAYOUT_GENERAL;
    } else {
      // Non-UAV images use SHADER_READ_ONLY for sampled access
      dxvkInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    if (desc.isRenderTarget) {
      dxvkInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      dxvkInfo.access |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }

    // Check format to determine if depth/stencil
    bool isDepth = (dxvkInfo.format >= VK_FORMAT_D16_UNORM && dxvkInfo.format <= VK_FORMAT_D32_SFLOAT_S8_UINT);
    if (isDepth) {
      dxvkInfo.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      dxvkInfo.access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      dxvkInfo.stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    }

    // Create DXVK image
    Rc<DxvkImage> dxvkImage = m_device->createImage(
      dxvkInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXMaterialTexture,
      desc.debugName ? desc.debugName : "NVRHI Texture");

    if (dxvkImage == nullptr) {
      Logger::err(str::format("Failed to create texture: ", desc.debugName ? desc.debugName : "unnamed"));
      return nullptr;
    }

    return new NvrhiDxvkTexture(desc, dxvkImage);
  }

  nvrhi::SamplerHandle NvrhiDxvkDevice::createSampler(const nvrhi::SamplerDesc& desc) {
    // Translate NVRHI sampler descriptor to DXVK sampler create info
    DxvkSamplerCreateInfo dxvkInfo;

    // Filter modes
    dxvkInfo.minFilter = desc.minFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    dxvkInfo.magFilter = desc.magFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    dxvkInfo.mipmapMode = desc.mipFilter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;

    // Address modes
    auto translateAddressMode = [](nvrhi::SamplerAddressMode mode) -> VkSamplerAddressMode {
      switch (mode) {
        case nvrhi::SamplerAddressMode::Clamp:       return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case nvrhi::SamplerAddressMode::Wrap:        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case nvrhi::SamplerAddressMode::Border:      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        case nvrhi::SamplerAddressMode::Mirror:      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case nvrhi::SamplerAddressMode::ClampToEdge: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        case nvrhi::SamplerAddressMode::MirrorOnce:  return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
        default:                                      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      }
    };

    dxvkInfo.addressModeU = translateAddressMode(desc.addressU);
    dxvkInfo.addressModeV = translateAddressMode(desc.addressV);
    dxvkInfo.addressModeW = translateAddressMode(desc.addressW);

    // Mip LOD bias and range
    dxvkInfo.mipmapLodBias = desc.mipBias;
    dxvkInfo.mipmapLodMin = 0.0f;
    dxvkInfo.mipmapLodMax = VK_LOD_CLAMP_NONE;

    // Anisotropy
    dxvkInfo.maxAnisotropy = desc.maxAnisotropy;
    dxvkInfo.useAnisotropy = (desc.maxAnisotropy > 1.0f) ? VK_TRUE : VK_FALSE;

    // Comparison/reduction type
    // Note: DXVK doesn't support sampler reduction modes, only comparison for depth
    switch (desc.reductionType) {
      case nvrhi::SamplerReductionType::Comparison:
        dxvkInfo.compareToDepth = VK_TRUE;
        dxvkInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        break;
      case nvrhi::SamplerReductionType::Minimum:
      case nvrhi::SamplerReductionType::Maximum:
      case nvrhi::SamplerReductionType::Standard:
      default:
        dxvkInfo.compareToDepth = VK_FALSE;
        dxvkInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        break;
    }

    // Border color (stored as VkClearColorValue union)
    if (desc.borderColor) {
      dxvkInfo.borderColor.float32[0] = 1.0f;
      dxvkInfo.borderColor.float32[1] = 1.0f;
      dxvkInfo.borderColor.float32[2] = 1.0f;
      dxvkInfo.borderColor.float32[3] = 1.0f;
    } else {
      dxvkInfo.borderColor.float32[0] = 0.0f;
      dxvkInfo.borderColor.float32[1] = 0.0f;
      dxvkInfo.borderColor.float32[2] = 0.0f;
      dxvkInfo.borderColor.float32[3] = 0.0f;
    }

    // Unnormalized coordinates - always use normalized for standard samplers
    dxvkInfo.usePixelCoord = VK_FALSE;

    // Create DXVK sampler
    Rc<DxvkSampler> dxvkSampler = m_device->createSampler(dxvkInfo);

    if (dxvkSampler == nullptr) {
      Logger::err("Failed to create sampler");
      return nullptr;
    }

    return new NvrhiDxvkSampler(desc, dxvkSampler);
  }

  nvrhi::ShaderHandle NvrhiDxvkDevice::createShader(
    const nvrhi::ShaderDesc& desc,
    const void* binary,
    size_t size)
  {
    // RTX MG shaders will be pre-compiled to SPIR-V and loaded via RTX Remix's shader system
    // This is a stub - actual shader loading happens in ClusterAccelBuilder via RtxShaderManager
    Logger::warn("NvrhiDxvkDevice::createShader() called - shaders should be loaded via RtxShaderManager");
    return nullptr;
  }

  nvrhi::ComputePipelineHandle NvrhiDxvkDevice::createComputePipeline(
    const nvrhi::ComputePipelineDesc& desc)
  {
    // Extract the DXVK shader from the NVRHI shader wrapper
    if (!desc.computeShader) {
      Logger::err("createComputePipeline: compute shader is null");
      return nullptr;
    }

    auto* shaderWrapper = static_cast<NvrhiDxvkShader*>(desc.computeShader.Get());
    if (!shaderWrapper) {
      Logger::err("createComputePipeline: failed to cast shader");
      return nullptr;
    }

    const Rc<DxvkShader>& dxvkShader = shaderWrapper->getDxvkShader();
    if (dxvkShader == nullptr) {
      Logger::err("createComputePipeline: DXVK shader is null");
      return nullptr;
    }

    // Create and return a compute pipeline wrapper
    return new NvrhiDxvkComputePipeline(dxvkShader, desc);
  }

  nvrhi::BindingLayoutHandle NvrhiDxvkDevice::createBindingLayout(
    const nvrhi::BindingLayoutDesc& desc)
  {
    // Create and return a binding layout wrapper that stores the description
    return new NvrhiDxvkBindingLayout(desc);
  }

  nvrhi::BindlessLayoutHandle NvrhiDxvkDevice::createBindlessLayout(
    const nvrhi::BindlessLayoutDesc& desc)
  {
    // Create a binding layout that represents the bindless descriptor space.
    // In RTX Remix, bindless resources are managed by the runtime, but we still
    // need a valid layout object so pipelines can be created correctly.
    // The actual bindless descriptors will be provided via createDescriptorTable().
    nvrhi::BindingLayoutDesc layoutDesc;
    layoutDesc.visibility = desc.visibility;
    // No explicit bindings - this is a bindless layout
    return new NvrhiDxvkBindingLayout(layoutDesc);
  }

  nvrhi::BindingSetHandle NvrhiDxvkDevice::createDescriptorTable(
    nvrhi::IBindingLayout* layout)
  {
    // Create an empty binding set for the bindless layout.
    // This serves as a placeholder descriptor table that satisfies the pipeline's
    // binding requirements. When displacement maps are enabled, this would need
    // to be populated with actual texture descriptors.
    nvrhi::BindingSetDesc emptyDesc;
    return new NvrhiDxvkBindingSet(emptyDesc, layout);
  }

  nvrhi::BindingSetHandle NvrhiDxvkDevice::createBindingSet(
    const nvrhi::BindingSetDesc& desc,
    nvrhi::IBindingLayout* layout)
  {
    // Create and return a binding set wrapper that stores the description and layout
    return new NvrhiDxvkBindingSet(desc, layout);
  }

  nvrhi::rt::cluster::OperationSizeInfo NvrhiDxvkDevice::getClusterOperationSizeInfo(
    const nvrhi::rt::cluster::OperationParams& params)
  {
    nvrhi::rt::cluster::OperationSizeInfo result = {};

    // Load the Vulkan function pointer for querying cluster AS sizes
    typedef void (VKAPI_PTR *PFN_vkGetClusterAccelerationStructureBuildSizesNV)(
      VkDevice device,
      const VkClusterAccelerationStructureInputInfoNV* pInfo,
      VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo);

    static PFN_vkGetClusterAccelerationStructureBuildSizesNV vkGetClusterASSizes = nullptr;
    if (!vkGetClusterASSizes) {
      vkGetClusterASSizes = reinterpret_cast<PFN_vkGetClusterAccelerationStructureBuildSizesNV>(
        m_device->vkd()->sym("vkGetClusterAccelerationStructureBuildSizesNV"));
      if (vkGetClusterASSizes) {
        Logger::info("RTX MegaGeo: Loaded vkGetClusterAccelerationStructureBuildSizesNV");
      }
    }

    if (!vkGetClusterASSizes) {
      Logger::err("RTX MegaGeo: vkGetClusterAccelerationStructureBuildSizesNV not available - using fallback estimates");
      // Fallback to conservative estimates
      result.resultMaxSizeInBytes = 256 * 1024 * 1024; // 256 MB
      result.scratchSizeInBytes = 256 * 1024 * 1024;   // 256 MB
      return result;
    }

    // Build the input info structure
    VkClusterAccelerationStructureInputInfoNV inputInfo = {};
    inputInfo.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV;
    inputInfo.maxAccelerationStructureCount = params.maxArgCount;

    // Map operation type
    switch (params.type) {
      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
        break;
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
        break;
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
        break;
      default:
        inputInfo.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
        break;
    }

    // Map operation mode
    switch (params.mode) {
      case nvrhi::rt::cluster::OperationMode::GetSizes:
        inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
        break;
      case nvrhi::rt::cluster::OperationMode::ExplicitDestinations:
        inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
        break;
      case nvrhi::rt::cluster::OperationMode::ImplicitDestinations:
        inputInfo.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
        break;
    }

    // Set up operation-specific input structures
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput = {};
    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput = {};

    switch (params.type) {
      case nvrhi::rt::cluster::OperationType::ClasBuildTemplates:
      case nvrhi::rt::cluster::OperationType::ClasInstantiateTemplates:
        triangleInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
        triangleInput.vertexFormat = static_cast<VkFormat>(params.clas.vertexFormat);
        triangleInput.maxGeometryIndexValue = params.clas.maxGeometryIndex;
        triangleInput.maxClusterUniqueGeometryCount = params.clas.maxUniqueGeometryCount;
        triangleInput.maxClusterTriangleCount = params.clas.maxTriangleCount;
        triangleInput.maxClusterVertexCount = params.clas.maxVertexCount;
        triangleInput.maxTotalTriangleCount = params.clas.maxTotalTriangleCount;
        triangleInput.maxTotalVertexCount = params.clas.maxTotalVertexCount;
        triangleInput.minPositionTruncateBitCount = params.clas.minPositionTruncateBitCount;
        inputInfo.opInput.pTriangleClusters = &triangleInput;
        break;
      case nvrhi::rt::cluster::OperationType::BlasBuild:
        blasInput.sType = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
        blasInput.maxClusterCountPerAccelerationStructure = params.blas.maxClasPerBlasCount;
        blasInput.maxTotalClusterCount = params.blas.maxTotalClasCount;
        inputInfo.opInput.pClustersBottomLevel = &blasInput;
        break;
      default:
        break;
    }

    // Query the actual sizes from the driver
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetClusterASSizes(m_device->handle(), &inputInfo, &sizeInfo);

    result.resultMaxSizeInBytes = sizeInfo.accelerationStructureSize;
    result.scratchSizeInBytes = sizeInfo.buildScratchSize;

    Logger::info(str::format("RTX MegaGeo: getClusterOperationSizeInfo - type=", (int)params.type,
      " resultMaxSize=", result.resultMaxSizeInBytes, " scratchSize=", result.scratchSizeInBytes));

    return result;
  }

  nvrhi::CommandListHandle NvrhiDxvkDevice::createCommandList(const nvrhi::CommandListParameters& params) {
    // Create a command list wrapper around the existing DxvkContext
    // In RTX Remix, we use a single shared context for most operations
    return new NvrhiDxvkCommandList(this, m_context);
  }

  uint64_t NvrhiDxvkDevice::executeCommandList(nvrhi::ICommandList* commandList) {
    // In RTX Remix/DXVK, command lists are executed immediately through DxvkContext
    // The command list wrapper (NvrhiDxvkCommandList) already submits commands to DxvkContext
    //
    // IMPORTANT: flushCommandList() just prepares a new command buffer, but doesn't
    // actually submit existing commands to GPU. The commands stay in DXVK's internal
    // queue until the frame is presented.
    //
    // For BuildAccel's synchronous readbacks to work, we need to actually submit
    // the commands to GPU and wait.

    if (m_context != nullptr) {
      // Flush pending commands to DXVK's internal queue
      // Note: We do NOT call waitForIdle() here because:
      // 1. DXVK's mapPtr() on staging buffers handles synchronization internally
      // 2. Calling waitForIdle() every frame kills performance
      m_context->flushCommandList();
    }

    // Return a dummy fence value (RTX Remix doesn't use explicit fences in the same way)
    return 1;
  }

  void NvrhiDxvkDevice::waitForIdle() {
    // Wait for all GPU work to complete.
    // IMPORTANT: Only call this when absolutely necessary (e.g., before destroying resources)
    // as it stalls the entire GPU pipeline and kills performance.
    m_device->waitForIdle();
  }

} // namespace dxvk
