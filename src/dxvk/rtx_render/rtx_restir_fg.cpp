/*
* Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_restir_fg.h"
#include "rtx_context.h"
#include "dxvk_device.h"
#include "rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_imgui.h"
#include "../dxvk_barrier.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"

#include <rtx_shaders/restir_fg_trace_photons.h>
#include <rtx_shaders/restir_fg_collect_photons.h>
#include <rtx_shaders/restir_fg_resample.h>
#include <rtx_shaders/restir_fg_caustic_resample.h>
#include <rtx_shaders/restir_fg_final_shading.h>

#include <array>

namespace dxvk {

  namespace {
    RemixGui::ComboWithKey<ReSTIRFGResamplingMode> resamplingModeCombo = RemixGui::ComboWithKey<ReSTIRFGResamplingMode>(
      "Resampling Mode",
      RemixGui::ComboWithKey<ReSTIRFGResamplingMode>::ComboEntries { {
        {ReSTIRFGResamplingMode::Temporal, "Temporal"},
        {ReSTIRFGResamplingMode::Spatial, "Spatial"},
        {ReSTIRFGResamplingMode::SpatioTemporal, "SpatioTemporal"},
      } });

    RemixGui::ComboWithKey<ReSTIRFGCausticMode> causticModeCombo = RemixGui::ComboWithKey<ReSTIRFGCausticMode>(
      "Caustic Mode",
      RemixGui::ComboWithKey<ReSTIRFGCausticMode>::ComboEntries { {
        {ReSTIRFGCausticMode::None, "None"},
        {ReSTIRFGCausticMode::Direct, "Direct"},
        {ReSTIRFGCausticMode::Temporal, "Temporal"},
        {ReSTIRFGCausticMode::Reservoir, "Reservoir"},
      } });

    RemixGui::ComboWithKey<ReSTIRFGBiasCorrectionMode> biasCorrectionModeCombo = RemixGui::ComboWithKey<ReSTIRFGBiasCorrectionMode>(
      "Bias Correction Mode",
      RemixGui::ComboWithKey<ReSTIRFGBiasCorrectionMode>::ComboEntries { {
        {ReSTIRFGBiasCorrectionMode::None, "None"},
        {ReSTIRFGBiasCorrectionMode::Basic, "Basic"},
        {ReSTIRFGBiasCorrectionMode::Raytraced, "Raytraced"},
        {ReSTIRFGBiasCorrectionMode::Pairwise, "Pairwise"},
      } });

    // Photon Tracing Shader
    class ReSTIRFGTracePhotonsShader : public ManagedShader {
      SHADER_SOURCE(ReSTIRFGTracePhotonsShader, VK_SHADER_STAGE_COMPUTE_BIT, restir_fg_trace_photons)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        RW_STRUCTURED_BUFFER(RESTIR_FG_TRACE_BINDING_PHOTON_BUFFER_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_TRACE_BINDING_PHOTON_AABB_GLOBAL_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_TRACE_BINDING_PHOTON_AABB_CAUSTIC_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_TRACE_BINDING_PHOTON_COUNTER)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ReSTIRFGTracePhotonsShader);

    // Photon Collection Shader
    class ReSTIRFGCollectPhotonsShader : public ManagedShader {
      SHADER_SOURCE(ReSTIRFGCollectPhotonsShader, VK_SHADER_STAGE_COMPUTE_BIT, restir_fg_collect_photons)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        // GBuffer inputs (matching ReSTIR GI pattern)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_SHARED_SURFACE_INDEX_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_PRIMARY_VIEW_DIRECTION_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_PRIMARY_CONE_RADIUS_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_POSITION_INPUT)
        TEXTURE2D(RESTIR_FG_COLLECT_BINDING_PRIMARY_POSITION_ERROR_INPUT)
        // Photon data
        ACCELERATION_STRUCTURE(RESTIR_FG_COLLECT_BINDING_PHOTON_AS)
        STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_PHOTON_DATA)
        STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_GLOBAL)
        STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_CAUSTIC)
        STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_PHOTON_COUNTER)
        // Outputs
        RW_STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_FG_RESERVOIR_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_FG_SAMPLE_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_CAUSTIC_RESERVOIR_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_CAUSTIC_SAMPLE_OUTPUT)
        RW_STRUCTURED_BUFFER(RESTIR_FG_COLLECT_BINDING_SURFACE_DATA_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ReSTIRFGCollectPhotonsShader);

    // Final Gather Resampling Shader
    class ReSTIRFGResampleShader : public ManagedShader {
      SHADER_SOURCE(ReSTIRFGResampleShader, VK_SHADER_STAGE_COMPUTE_BIT, restir_fg_resample)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        // Inputs
        TEXTURE2D(RESTIR_FG_RESAMPLE_BINDING_MVEC_INPUT)
        TEXTURE2D(RESTIR_FG_RESAMPLE_BINDING_WORLD_POSITION_INPUT)
        TEXTURE2D(RESTIR_FG_RESAMPLE_BINDING_WORLD_NORMAL_INPUT)
        STRUCTURED_BUFFER(RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_PREV)
        STRUCTURED_BUFFER(RESTIR_FG_RESAMPLE_BINDING_SAMPLE_PREV)
        STRUCTURED_BUFFER(RESTIR_FG_RESAMPLE_BINDING_SURFACE_PREV)
        STRUCTURED_BUFFER(RESTIR_FG_RESAMPLE_BINDING_SURFACE_CURR)
        // Inputs / Outputs
        RW_STRUCTURED_BUFFER(RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_CURR)
        RW_STRUCTURED_BUFFER(RESTIR_FG_RESAMPLE_BINDING_SAMPLE_CURR)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ReSTIRFGResampleShader);

    // Caustic Resampling Shader
    class ReSTIRFGCausticResampleShader : public ManagedShader {
      SHADER_SOURCE(ReSTIRFGCausticResampleShader, VK_SHADER_STAGE_COMPUTE_BIT, restir_fg_caustic_resample)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        TEXTURE2D(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_MVEC_INPUT)
        TEXTURE2D(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_POSITION_INPUT)
        TEXTURE2D(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_NORMAL_INPUT)
        STRUCTURED_BUFFER(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_PREV)
        STRUCTURED_BUFFER(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_PREV)
        STRUCTURED_BUFFER(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SURFACE_PREV)
        STRUCTURED_BUFFER(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SURFACE_CURR)
        RW_STRUCTURED_BUFFER(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_CURR)
        RW_STRUCTURED_BUFFER(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_CURR)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ReSTIRFGCausticResampleShader);

    // Final Shading Shader
    class ReSTIRFGFinalShadingShader : public ManagedShader {
      SHADER_SOURCE(ReSTIRFGFinalShadingShader, VK_SHADER_STAGE_COMPUTE_BIT, restir_fg_final_shading)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS
        // Inputs
        TEXTURE2D(RESTIR_FG_FINAL_SHADING_BINDING_WORLD_POSITION_INPUT)
        TEXTURE2D(RESTIR_FG_FINAL_SHADING_BINDING_WORLD_NORMAL_INPUT)
        TEXTURE2D(RESTIR_FG_FINAL_SHADING_BINDING_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(RESTIR_FG_FINAL_SHADING_BINDING_ALBEDO_INPUT)
        STRUCTURED_BUFFER(RESTIR_FG_FINAL_SHADING_BINDING_FG_RESERVOIR)
        STRUCTURED_BUFFER(RESTIR_FG_FINAL_SHADING_BINDING_FG_SAMPLE)
        STRUCTURED_BUFFER(RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_RESERVOIR)
        STRUCTURED_BUFFER(RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_SAMPLE)
        // Outputs
        RW_TEXTURE2D(RESTIR_FG_FINAL_SHADING_BINDING_OUTPUT)
        RW_TEXTURE2D(RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ReSTIRFGFinalShadingShader);

    // Must match packed structs in restir_fg_types.slangh
    constexpr uint32_t kFGReservoirSize = 16;
    constexpr uint32_t kCausticReservoirSize = 16;
    constexpr uint32_t kFGSampleSize = 64;
    constexpr uint32_t kCausticSampleSize = 48;
    constexpr uint32_t kSurfaceDataSize = 48;
  }

  DxvkReSTIRFG::DxvkReSTIRFG(DxvkDevice* device) : RtxPass(device) {
  }

  bool DxvkReSTIRFG::isEnabled() const {
    return RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::ReSTIRFG;
  }

  void DxvkReSTIRFG::showImguiSettings() {

    if (ImGui::CollapsingHeader("Photon Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      RemixGui::DragInt("Photons Per Frame", reinterpret_cast<int*>(&photonsPerFrameObject()), 1000.f, 10000, 4000000, "%d");
      RemixGui::DragInt("Max Photon Bounces", reinterpret_cast<int*>(&maxPhotonBouncesObject()), 1.f, 1, 32, "%d");
      RemixGui::DragFloat("Global Photon Radius", &photonRadiusObject(), 0.001f, 0.001f, 1.0f, "%.4f");
      RemixGui::DragFloat("Caustic Photon Radius", &causticPhotonRadiusObject(), 0.001f, 0.001f, 0.5f, "%.4f");
      RemixGui::DragFloat("Global Rejection Probability", &globalPhotonRejectionProbabilityObject(), 0.01f, 0.0f, 0.99f, "%.2f");
      RemixGui::DragFloat("Roughness Threshold", &roughnessThresholdObject(), 0.01f, 0.01f, 1.0f, "%.2f");
    }

    if (ImGui::CollapsingHeader("Final Gather Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      RemixGui::DragInt("Max FG Bounces", reinterpret_cast<int*>(&maxFinalGatherBouncesObject()), 1.f, 1, 16, "%d");
      RemixGui::Checkbox("Use FG RIS", &useFinalGatherRISObject());
    }

    if (ImGui::CollapsingHeader("Resampling Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      resamplingModeCombo.getKey(&resamplingModeObject());
      RemixGui::DragInt("Temporal History Length", reinterpret_cast<int*>(&temporalHistoryLengthObject()), 1.f, 1, 100, "%d");
      RemixGui::DragInt("Spatial Samples", reinterpret_cast<int*>(&spatialSamplesObject()), 1.f, 1, 16, "%d");
      RemixGui::DragFloat("Spatial Radius", &spatialRadiusObject(), 0.5f, 1.0f, 100.0f, "%.1f");
      RemixGui::DragInt("Disocclusion Boost Samples", reinterpret_cast<int*>(&disocclusionBoostSamplesObject()), 1.f, 0, 32, "%d");
      RemixGui::DragFloat("Normal Threshold", &normalThresholdObject(), 0.01f, 0.5f, 1.0f, "%.2f");
      RemixGui::DragFloat("Depth Threshold", &depthThresholdObject(), 0.01f, 0.01f, 0.5f, "%.2f");
    }

    if (ImGui::CollapsingHeader("Caustic Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      causticModeCombo.getKey(&causticModeObject());
      if (causticMode() != ReSTIRFGCausticMode::None) {
        RemixGui::DragInt("Caustic Spatial Samples", reinterpret_cast<int*>(&causticSpatialSamplesObject()), 1.f, 1, 8, "%d");
        // Very small radius should be used - limit to 10 pixels max
        RemixGui::DragFloat("Caustic Spatial Radius", &causticSpatialRadiusObject(), 0.25f, 0.5f, 10.0f, "%.1f");
      }
    }

    if (ImGui::CollapsingHeader("Bias Correction", ImGuiTreeNodeFlags_DefaultOpen)) {
      biasCorrectionModeCombo.getKey(&biasCorrectionModeObject());
      if (biasCorrectionMode() == ReSTIRFGBiasCorrectionMode::Pairwise) {
        RemixGui::DragFloat("Pairwise MIS Central Weight", &pairwiseMISCentralWeightObject(), 0.01f, 0.01f, 1.0f, "%.2f");
      }
      RemixGui::DragFloat("Max Luminance", &maxLuminanceObject(), 0.5f, 1.0f, 100.0f, "%.1f");
      RemixGui::DragFloat("Min Photon Contribution", &minPhotonContributionObject(), 0.0001f, 0.0001f, 0.1f, "%.4f");
    }

    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
      RemixGui::Checkbox("Use Photon Culling", &usePhotonCullingObject());
      RemixGui::Checkbox("Use Split Collection", &useSplitCollectionObject());
      RemixGui::Checkbox("Use Stochastic Collection", &useStochasticCollectionObject());
    }
  }

  void DxvkReSTIRFG::setToNRDPreset() {
    // Optimized settings for NRD denoising
    spatialSamplesObject().setImmediately(2);
    spatialRadiusObject().setImmediately(12.0f);
    temporalHistoryLengthObject().setImmediately(30);
  }

  void DxvkReSTIRFG::setToRayReconstructionPreset() {
    // Optimized settings for DLSS Ray Reconstruction
    spatialSamplesObject().setImmediately(2);
    spatialRadiusObject().setImmediately(14.0f);
    temporalHistoryLengthObject().setImmediately(30);
    biasCorrectionModeObject().setImmediately(ReSTIRFGBiasCorrectionMode::Pairwise);
  }

  void DxvkReSTIRFG::createDownscaledResource(Rc<DxvkContext>& ctx, const VkExtent3D& downscaledExtent) {
    const uint32_t pixelCount = downscaledExtent.width * downscaledExtent.height;

    // Photon buffer - stores photon data (position, direction, flux)
    // Each photon: position (3 floats) + direction (packed) + flux (3 floats) = 32 bytes
    const uint32_t maxPhotons = photonsPerFrame() * 2; // Double buffer for global + caustic
    DxvkBufferCreateInfo photonBufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    photonBufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    photonBufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    photonBufferInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    photonBufferInfo.size = maxPhotons * 32; // PhotonData struct size
    m_photonBuffer = ctx->getDevice()->createBuffer(photonBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Photon Buffer");

    // Photon AABB buffers for acceleration structure
    DxvkBufferCreateInfo aabbBufferInfo = photonBufferInfo;
    aabbBufferInfo.usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    aabbBufferInfo.size = maxPhotons * sizeof(VkAabbPositionsKHR); // 6 floats (24 bytes) per AABB
    m_photonAABBBuffer[0] = ctx->getDevice()->createBuffer(aabbBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Global Photon AABB");
    m_photonAABBBuffer[1] = ctx->getDevice()->createBuffer(aabbBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Caustic Photon AABB");

    // Photon counter buffer
    DxvkBufferCreateInfo counterBufferInfo = photonBufferInfo;
    counterBufferInfo.size = sizeof(uint32_t) * 4; // [globalCount, causticCount, padding, padding]
    m_photonCounterBuffer = ctx->getDevice()->createBuffer(counterBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Photon Counter");

    // Reservoir buffers (double-buffered)
    // FG Reservoir: weightSum + targetFunc + M + age = 16 bytes packed
    const uint32_t reservoirSize = kFGReservoirSize;
    DxvkBufferCreateInfo reservoirBufferInfo = photonBufferInfo;
    reservoirBufferInfo.size = pixelCount * 2 * reservoirSize; // Double buffer
    m_fgReservoirBuffer = ctx->getDevice()->createBuffer(reservoirBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG FG Reservoir Buffer");
    m_causticReservoirBuffer = ctx->getDevice()->createBuffer(reservoirBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Caustic Reservoir Buffer");

    // Sample data buffers
    // FG Sample: PackedReSTIRFG_Sample = 64 bytes
    const uint32_t fgSampleSize = kFGSampleSize;
    DxvkBufferCreateInfo sampleBufferInfo = photonBufferInfo;
    sampleBufferInfo.size = pixelCount * 2 * fgSampleSize;
    m_fgSampleBuffer = ctx->getDevice()->createBuffer(sampleBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG FG Sample Buffer");
    
    // Caustic Sample: PackedReSTIRFG_CausticSample = 48 bytes
    const uint32_t causticSampleSize = kCausticSampleSize;
    sampleBufferInfo.size = pixelCount * 2 * causticSampleSize;
    m_causticSampleBuffer = ctx->getDevice()->createBuffer(sampleBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Caustic Sample Buffer");

    // Surface data buffer
    // Surface: position + normal + roughness + albedo = 48 bytes
    const uint32_t surfaceSize = kSurfaceDataSize;
    sampleBufferInfo.size = pixelCount * 2 * surfaceSize;
    m_surfaceBuffer = ctx->getDevice()->createBuffer(sampleBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR-FG Surface Buffer");

    // Output textures
    m_fgRadiance = Resources::createImageResource(ctx, "ReSTIR-FG FG Radiance", downscaledExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_causticRadiance = Resources::createImageResource(ctx, "ReSTIR-FG Caustic Radiance", downscaledExtent, VK_FORMAT_R16G16B16A16_SFLOAT);

    const Resources::RaytracingOutput& rtOutput = ctx->getCommonObjects()->getResources().getRaytracingOutput();
    m_combinedRadiance = Resources::AliasedResource(rtOutput.m_compositeOutput, ctx, downscaledExtent, VK_FORMAT_R16G16B16A16_SFLOAT, "ReSTIR-FG Combined");
  }

  void DxvkReSTIRFG::releaseDownscaledResource() {
    m_photonBuffer = nullptr;
    m_photonAABBBuffer[0] = nullptr;
    m_photonAABBBuffer[1] = nullptr;
    m_photonCounterBuffer = nullptr;
    m_photonASBuffer = nullptr;
    m_photonAccelStructure = nullptr;
    m_photonScratchBuffer = nullptr;
    m_fgReservoirBuffer = nullptr;
    m_causticReservoirBuffer = nullptr;
    m_fgSampleBuffer = nullptr;
    m_causticSampleBuffer = nullptr;
    m_surfaceBuffer = nullptr;
    m_fgRadiance.reset();
    m_causticRadiance.reset();
    m_combinedRadiance.reset();
    m_photonAS.reset();
  }

  void DxvkReSTIRFG::dispatch(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    if (!isActive()) {
      return;
    }

    ScopedGpuProfileZone(ctx, "ReSTIR-FG");

    // Step 1: Trace photons from light sources
    tracePhotons(ctx, rtOutput);

    // Step 2: Build photon acceleration structure
    buildPhotonAccelerationStructure(ctx);

    // Step 3: Collect photons at surfaces and initialize reservoirs
    collectPhotons(ctx, rtOutput);

    // Step 4: Resample Final Gather reservoirs (temporal, spatial, or both)
    if (m_canResample) {
      resampleFinalGather(ctx, rtOutput);
    }

    // Step 5: Resample Caustic reservoirs (temporal-only or full spatiotemporal)
    if (m_canResample && (causticMode() == ReSTIRFGCausticMode::Temporal || causticMode() == ReSTIRFGCausticMode::Reservoir)) {
      resampleCaustics(ctx, rtOutput);
    }

    // Step 6: Final shading - evaluate reservoirs
    finalShading(ctx, rtOutput);

    // Update frame tracking
    m_frameCount++;
    m_canResample = true;
  }

  void DxvkReSTIRFG::tracePhotons(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Trace Photons");

    // Clear photon counter
    ctx->clearBuffer(m_photonCounterBuffer, 0, m_photonCounterBuffer->info().size, 0);

    ctx->bindCommonRayTracingResources(rtOutput);

    // Bind outputs
    ctx->bindResourceBuffer(RESTIR_FG_TRACE_BINDING_PHOTON_BUFFER_OUTPUT, DxvkBufferSlice(m_photonBuffer, 0, m_photonBuffer->info().size));
    ctx->bindResourceBuffer(RESTIR_FG_TRACE_BINDING_PHOTON_AABB_GLOBAL_OUTPUT, DxvkBufferSlice(m_photonAABBBuffer[0], 0, m_photonAABBBuffer[0]->info().size));
    ctx->bindResourceBuffer(RESTIR_FG_TRACE_BINDING_PHOTON_AABB_CAUSTIC_OUTPUT, DxvkBufferSlice(m_photonAABBBuffer[1], 0, m_photonAABBBuffer[1]->info().size));
    ctx->bindResourceBuffer(RESTIR_FG_TRACE_BINDING_PHOTON_COUNTER, DxvkBufferSlice(m_photonCounterBuffer, 0, m_photonCounterBuffer->info().size));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ReSTIRFGTracePhotonsShader::getShader());

    // Dispatch photon tracing - use 1D dispatch for photon emission
    const uint32_t photonWorkgroups = (photonsPerFrame() + 255) / 256;
    ctx->dispatch(photonWorkgroups, 1, 1);
  }

  void DxvkReSTIRFG::buildPhotonAccelerationStructure(RtxContext* ctx) {
    ScopedGpuProfileZone(ctx, "Build Photon AS");
    
    // Build BLAS from photon AABBs for efficient photon lookup
    // We create two geometry instances: global photons and caustic photons
    
    const uint32_t maxPhotonsPerType = photonsPerFrame();
    
    // Set up geometry for global photons
    VkAccelerationStructureGeometryKHR geometryGlobal {};
    geometryGlobal.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometryGlobal.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometryGlobal.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometryGlobal.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
    geometryGlobal.geometry.aabbs.data.deviceAddress = m_photonAABBBuffer[0]->getDeviceAddress();
    geometryGlobal.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    
    // Set up geometry for caustic photons
    VkAccelerationStructureGeometryKHR geometryCaustic {};
    geometryCaustic.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometryCaustic.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometryCaustic.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometryCaustic.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
    geometryCaustic.geometry.aabbs.data.deviceAddress = m_photonAABBBuffer[1]->getDeviceAddress();
    geometryCaustic.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    
    std::array<VkAccelerationStructureGeometryKHR, 2> geometries = { geometryGlobal, geometryCaustic };
    
    // Build info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.pNext = nullptr;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount = 2;
    buildInfo.pGeometries = geometries.data();
    buildInfo.ppGeometries = nullptr;
    
    // Get size requirements
    std::array<uint32_t, 2> maxPrimitiveCounts = { maxPhotonsPerType, maxPhotonsPerType };
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    
    DxvkDevice* device = ctx->getDevice().ptr();
    device->vkd()->vkGetAccelerationStructureBuildSizesKHR(
      device->handle(), 
      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, 
      &buildInfo, 
      maxPrimitiveCounts.data(), 
      &sizeInfo);
    
    // Create or resize acceleration structure buffer if needed
    if (!m_photonAS.image.ptr() || m_photonASBuffer == nullptr || 
        m_photonASBuffer->info().size < sizeInfo.accelerationStructureSize) {
      
      DxvkBufferCreateInfo asBufferInfo {};
      asBufferInfo.size = sizeInfo.accelerationStructureSize;
      asBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      asBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      asBufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      
      m_photonASBuffer = device->createBuffer(asBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
        DxvkMemoryStats::Category::RTXAccelerationStructure, "ReSTIR-FG Photon AS Buffer");
      
      m_photonAccelStructure = device->createAccelStructure(asBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, "ReSTIR-FG Photon BLAS");
    }
    
    buildInfo.dstAccelerationStructure = m_photonAccelStructure->getAccelStructure();
    
    // Create scratch buffer if needed
    const VkDeviceSize scratchAlignment = device->properties().khrDeviceAccelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment;
    const VkDeviceSize requiredScratchSize = sizeInfo.buildScratchSize + scratchAlignment;
    
    if (m_photonScratchBuffer == nullptr || m_photonScratchBuffer->info().size < requiredScratchSize) {
      DxvkBufferCreateInfo scratchBufferInfo {};
      scratchBufferInfo.size = requiredScratchSize;
      scratchBufferInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      scratchBufferInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      scratchBufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      
      m_photonScratchBuffer = device->createBuffer(scratchBufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXAccelerationStructure, "ReSTIR-FG Photon Scratch");
    }
    
    // Align scratch buffer address
    VkDeviceAddress scratchAddress = m_photonScratchBuffer->getDeviceAddress();
    scratchAddress = (scratchAddress + scratchAlignment - 1) & ~(scratchAlignment - 1);
    buildInfo.scratchData.deviceAddress = scratchAddress;
    
    // Memory barrier before build
    DxvkBarrierSet barriers(DxvkCmdBuffer::ExecBuffer);
    
    barriers.accessBuffer(
      m_photonAABBBuffer[0]->getSliceHandle(),
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);
    
    barriers.accessBuffer(
      m_photonAABBBuffer[1]->getSliceHandle(),
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);
    
    barriers.recordCommands(ctx->getCommandList());
    
    // Build ranges - use actual photon counts from counter buffer
    // For now, use max counts since we don't have a way to read back the counter synchronously
    std::array<VkAccelerationStructureBuildRangeInfoKHR, 2> buildRanges {};
    buildRanges[0].primitiveCount = maxPhotonsPerType;
    buildRanges[0].primitiveOffset = 0;
    buildRanges[0].firstVertex = 0;
    buildRanges[0].transformOffset = 0;
    
    buildRanges[1].primitiveCount = maxPhotonsPerType;
    buildRanges[1].primitiveOffset = 0;
    buildRanges[1].firstVertex = 0;
    buildRanges[1].transformOffset = 0;
    
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRanges = buildRanges.data();
    
    ctx->getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildRanges);
    
    // Track resources
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_photonAABBBuffer[0]);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_photonAABBBuffer[1]);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_photonScratchBuffer);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(m_photonASBuffer);
    
    // Memory barrier after build
    barriers.accessBuffer(
      m_photonASBuffer->getSliceHandle(),
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    
    barriers.recordCommands(ctx->getCommandList());
  }

  void DxvkReSTIRFG::collectPhotons(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Collect Photons");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    // Bind GBuffer inputs (matching ReSTIR GI pattern)
    // Note: Resource types use .view (member), AliasedResource types use .view(AccessType) (method)
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_SHARED_FLAGS_INPUT, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_SHARED_SURFACE_INDEX_INPUT, rtOutput.m_sharedSurfaceIndex.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT, rtOutput.m_primaryPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_PRIMARY_VIEW_DIRECTION_INPUT, rtOutput.m_primaryViewDirection.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_PRIMARY_CONE_RADIUS_INPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(RESTIR_FG_COLLECT_BINDING_PRIMARY_POSITION_ERROR_INPUT, rtOutput.m_primaryPositionError.view, nullptr);

    // Bind photon acceleration structure
    ctx->bindAccelerationStructure(RESTIR_FG_COLLECT_BINDING_PHOTON_AS, m_photonAccelStructure);
    
    // Bind photon data
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_PHOTON_DATA, DxvkBufferSlice(m_photonBuffer, 0, m_photonBuffer->info().size));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_GLOBAL, DxvkBufferSlice(m_photonAABBBuffer[0], 0, m_photonAABBBuffer[0]->info().size));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_CAUSTIC, DxvkBufferSlice(m_photonAABBBuffer[1], 0, m_photonAABBBuffer[1]->info().size));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_PHOTON_COUNTER, DxvkBufferSlice(m_photonCounterBuffer, 0, m_photonCounterBuffer->info().size));

    // Bind outputs
    const uint32_t frameIdx = m_frameCount % 2;
    const uint32_t pixelCount = numRaysExtent.width * numRaysExtent.height;
    
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_FG_RESERVOIR_OUTPUT, DxvkBufferSlice(m_fgReservoirBuffer, frameIdx * pixelCount * kFGReservoirSize, pixelCount * kFGReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_FG_SAMPLE_OUTPUT, DxvkBufferSlice(m_fgSampleBuffer, frameIdx * pixelCount * kFGSampleSize, pixelCount * kFGSampleSize));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_CAUSTIC_RESERVOIR_OUTPUT, DxvkBufferSlice(m_causticReservoirBuffer, frameIdx * pixelCount * kCausticReservoirSize, pixelCount * kCausticReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_CAUSTIC_SAMPLE_OUTPUT, DxvkBufferSlice(m_causticSampleBuffer, frameIdx * pixelCount * kCausticSampleSize, pixelCount * kCausticSampleSize));
    ctx->bindResourceBuffer(RESTIR_FG_COLLECT_BINDING_SURFACE_DATA_OUTPUT, DxvkBufferSlice(m_surfaceBuffer, frameIdx * pixelCount * kSurfaceDataSize, pixelCount * kSurfaceDataSize));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ReSTIRFGCollectPhotonsShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkReSTIRFG::resampleFinalGather(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Resample FG");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    const uint32_t currIdx = m_frameCount % 2;
    const uint32_t prevIdx = (m_frameCount + 1) % 2;
    const uint32_t pixelCount = numRaysExtent.width * numRaysExtent.height;

    uint32_t reservoirCurrIdx = currIdx;
    uint32_t reservoirPrevIdx = prevIdx;
    if (resamplingMode() == ReSTIRFGResamplingMode::Spatial) {
      std::swap(reservoirCurrIdx, reservoirPrevIdx);
    }

    const uint32_t surfaceCurrIdx = currIdx;
    const uint32_t surfacePrevIdx = prevIdx;

    // Bind inputs
    ctx->bindResourceView(RESTIR_FG_RESAMPLE_BINDING_MVEC_INPUT, rtOutput.m_primaryScreenSpaceMotionVector.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_RESAMPLE_BINDING_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(RESTIR_FG_RESAMPLE_BINDING_WORLD_NORMAL_INPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);

    ctx->bindResourceBuffer(RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_PREV, DxvkBufferSlice(m_fgReservoirBuffer, reservoirPrevIdx * pixelCount * kFGReservoirSize, pixelCount * kFGReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_RESAMPLE_BINDING_SAMPLE_PREV, DxvkBufferSlice(m_fgSampleBuffer, reservoirPrevIdx * pixelCount * kFGSampleSize, pixelCount * kFGSampleSize));
    ctx->bindResourceBuffer(RESTIR_FG_RESAMPLE_BINDING_SURFACE_PREV, DxvkBufferSlice(m_surfaceBuffer, surfacePrevIdx * pixelCount * kSurfaceDataSize, pixelCount * kSurfaceDataSize));
    ctx->bindResourceBuffer(RESTIR_FG_RESAMPLE_BINDING_SURFACE_CURR, DxvkBufferSlice(m_surfaceBuffer, surfaceCurrIdx * pixelCount * kSurfaceDataSize, pixelCount * kSurfaceDataSize));

    // Bind outputs
    ctx->bindResourceBuffer(RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_CURR, DxvkBufferSlice(m_fgReservoirBuffer, reservoirCurrIdx * pixelCount * kFGReservoirSize, pixelCount * kFGReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_RESAMPLE_BINDING_SAMPLE_CURR, DxvkBufferSlice(m_fgSampleBuffer, reservoirCurrIdx * pixelCount * kFGSampleSize, pixelCount * kFGSampleSize));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ReSTIRFGResampleShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkReSTIRFG::resampleCaustics(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Resample Caustics");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    const uint32_t currIdx = m_frameCount % 2;
    const uint32_t prevIdx = (m_frameCount + 1) % 2;
    const uint32_t pixelCount = numRaysExtent.width * numRaysExtent.height;

    const uint32_t reservoirCurrIdx = currIdx;
    const uint32_t reservoirPrevIdx = prevIdx;
    const uint32_t surfaceCurrIdx = currIdx;
    const uint32_t surfacePrevIdx = prevIdx;

    // Bind inputs
    ctx->bindResourceView(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_MVEC_INPUT, rtOutput.m_primaryScreenSpaceMotionVector.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_WORLD_NORMAL_INPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);

    ctx->bindResourceBuffer(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_PREV, DxvkBufferSlice(m_causticReservoirBuffer, reservoirPrevIdx * pixelCount * kCausticReservoirSize, pixelCount * kCausticReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_PREV, DxvkBufferSlice(m_causticSampleBuffer, reservoirPrevIdx * pixelCount * kCausticSampleSize, pixelCount * kCausticSampleSize));
    ctx->bindResourceBuffer(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SURFACE_PREV, DxvkBufferSlice(m_surfaceBuffer, surfacePrevIdx * pixelCount * kSurfaceDataSize, pixelCount * kSurfaceDataSize));
    ctx->bindResourceBuffer(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SURFACE_CURR, DxvkBufferSlice(m_surfaceBuffer, surfaceCurrIdx * pixelCount * kSurfaceDataSize, pixelCount * kSurfaceDataSize));

    // Bind outputs
    ctx->bindResourceBuffer(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_RESERVOIR_CURR, DxvkBufferSlice(m_causticReservoirBuffer, reservoirCurrIdx * pixelCount * kCausticReservoirSize, pixelCount * kCausticReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_CAUSTIC_RESAMPLE_BINDING_SAMPLE_CURR, DxvkBufferSlice(m_causticSampleBuffer, reservoirCurrIdx * pixelCount * kCausticSampleSize, pixelCount * kCausticSampleSize));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ReSTIRFGCausticResampleShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkReSTIRFG::finalShading(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    ScopedGpuProfileZone(ctx, "Final Shading");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    const uint32_t currIdx = m_frameCount % 2;
    const uint32_t prevIdx = (m_frameCount + 1) % 2;
    const bool fgSpatialResampled = m_canResample && (resamplingMode() == ReSTIRFGResamplingMode::Spatial);
    const uint32_t fgReservoirIdx = fgSpatialResampled ? prevIdx : currIdx;
    const uint32_t pixelCount = numRaysExtent.width * numRaysExtent.height;

    // Bind inputs
    ctx->bindResourceView(RESTIR_FG_FINAL_SHADING_BINDING_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(RESTIR_FG_FINAL_SHADING_BINDING_WORLD_NORMAL_INPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_FINAL_SHADING_BINDING_PERCEPTUAL_ROUGHNESS_INPUT, rtOutput.m_primaryPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(RESTIR_FG_FINAL_SHADING_BINDING_ALBEDO_INPUT, rtOutput.m_primaryAlbedo.view, nullptr);

    ctx->bindResourceBuffer(RESTIR_FG_FINAL_SHADING_BINDING_FG_RESERVOIR, DxvkBufferSlice(m_fgReservoirBuffer, fgReservoirIdx * pixelCount * kFGReservoirSize, pixelCount * kFGReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_FINAL_SHADING_BINDING_FG_SAMPLE, DxvkBufferSlice(m_fgSampleBuffer, fgReservoirIdx * pixelCount * kFGSampleSize, pixelCount * kFGSampleSize));
    ctx->bindResourceBuffer(RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_RESERVOIR, DxvkBufferSlice(m_causticReservoirBuffer, currIdx * pixelCount * kCausticReservoirSize, pixelCount * kCausticReservoirSize));
    ctx->bindResourceBuffer(RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_SAMPLE, DxvkBufferSlice(m_causticSampleBuffer, currIdx * pixelCount * kCausticSampleSize, pixelCount * kCausticSampleSize));

    // Bind outputs - write directly to primary indirect radiance for compositor integration
    ctx->bindResourceView(RESTIR_FG_FINAL_SHADING_BINDING_OUTPUT, rtOutput.m_primaryIndirectDiffuseRadiance.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_OUTPUT, rtOutput.m_primaryIndirectSpecularRadiance.view(Resources::AccessType::Write), nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ReSTIRFGFinalShadingShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }
}
