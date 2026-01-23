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
#pragma once

#include "../dxvk_format.h"
#include "../dxvk_include.h"
#include "../dxvk_buffer.h"

#include "../spirv/spirv_code_buffer.h"
#include "rtx_resources.h"
#include "rtx_options.h"

namespace dxvk {

  // ReSTIR-FG: Real-Time Reservoir Resampled Photon Final Gathering
  // Based on the paper by Kern, Brüll, and Grosch (TU Clausthal)
  // https://diglib.eg.org/items/df98f89d-a0ca-4800-9bc4-74528feaf872

  enum class ReSTIRFGResamplingMode : int {
    Temporal,
    Spatial,
    SpatioTemporal
  };

  enum class ReSTIRFGCausticMode : int {
    None,           // Disable caustic photon collection
    Direct,         // Direct collection (no resampling)
    Temporal,       // Temporal resampling only
    Reservoir       // Full reservoir resampling (spatiotemporal)
  };

  enum class ReSTIRFGBiasCorrectionMode : int {
    None,
    Basic,
    Raytraced,
    Pairwise
  };

  class RtxContext;

  class DxvkReSTIRFG : public RtxPass {
  public:
    DxvkReSTIRFG(DxvkDevice* device);
    ~DxvkReSTIRFG() = default;

    void dispatch(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    void showImguiSettings();

    const Resources::Resource& getPhotonAccelerationStructure() const { return m_photonAS; }
    const Rc<DxvkBuffer>& getPhotonBuffer() const { return m_photonBuffer; }
    
    static void setToNRDPreset();
    static void setToRayReconstructionPreset();

  private:
    virtual bool isEnabled() const override;
    virtual void releaseDownscaledResource() override;
    virtual void createDownscaledResource(Rc<DxvkContext>& ctx, const VkExtent3D& downscaledExtent) override;

    void tracePhotons(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);
    void buildPhotonAccelerationStructure(RtxContext* ctx);
    void collectPhotons(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);
    void resampleFinalGather(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);
    void resampleCaustics(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);
    void finalShading(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    // --- Configuration Options ---

    // Enable/Disable
    RTX_OPTION("rtx.restirFG", bool, enable, false, "Enables ReSTIR Final Gathering for global illumination. This uses photon mapping combined with reservoir resampling for real-time GI and caustics.");

    // Photon Settings
    RTX_OPTION("rtx.restirFG", uint32_t, photonsPerFrame, 2000000, "Number of photons to trace per frame. Higher values improve quality but reduce performance.");
    RTX_OPTION("rtx.restirFG", uint32_t, maxPhotonBounces, 10, "Maximum number of bounces for photon tracing.");
    RTX_OPTION("rtx.restirFG", float, photonRadius, 0.02f, "Base collection radius for global photons. Automatically adjusted based on scene extent.");
    RTX_OPTION("rtx.restirFG", float, causticPhotonRadius, 0.005f, "Collection radius for caustic photons. Smaller radius for sharper caustics.");
    RTX_OPTION("rtx.restirFG", float, globalPhotonRejectionProbability, 0.3f, "Probability of rejecting global photons to increase relative caustic density. Range [0,1].");
    RTX_OPTION("rtx.restirFG", float, roughnessThreshold, 0.25f, "Roughness threshold to classify surfaces as diffuse (for photon storage).");

    // Final Gather Settings
    RTX_OPTION("rtx.restirFG", uint32_t, maxFinalGatherBounces, 10, "Maximum path length for final gather samples.");
    RTX_OPTION("rtx.restirFG", bool, useFinalGatherRIS, true, "Use Resampled Importance Sampling for final gather samples.");

    // Resampling Settings
    RTX_OPTION("rtx.restirFG", ReSTIRFGResamplingMode, resamplingMode, ReSTIRFGResamplingMode::SpatioTemporal, "Resampling mode for final gather reservoirs.");
    RTX_OPTION("rtx.restirFG", uint32_t, temporalHistoryLength, 20, "Maximum temporal history length for reservoir resampling.");
    RTX_OPTION("rtx.restirFG", uint32_t, spatialSamples, 1, "Number of spatial samples for reservoir resampling.");
    RTX_OPTION("rtx.restirFG", float, spatialRadius, 20.0f, "Pixel radius for spatial resampling.");
    RTX_OPTION("rtx.restirFG", uint32_t, disocclusionBoostSamples, 2, "Extra spatial samples when temporal resampling fails.");
    RTX_OPTION("rtx.restirFG", float, normalThreshold, 0.6f, "Cosine threshold for normal similarity check in resampling.");
    RTX_OPTION("rtx.restirFG", float, depthThreshold, 0.15f, "Relative depth threshold for similarity check in resampling.");

    // Caustic Settings
    // For caustic reservoirs, temporal resampling is mostly sufficient.
    // If spatial resampling is used, a very small radius should be used to slightly improve quality in motion.
    RTX_OPTION("rtx.restirFG", ReSTIRFGCausticMode, causticMode, ReSTIRFGCausticMode::Reservoir, "Mode for caustic photon collection and resampling. Reservoir (spatiotemporal) is recommended.");
    RTX_OPTION("rtx.restirFG", uint32_t, causticSpatialSamples, 1, "Number of spatial samples for caustic reservoir resampling. Keep very small.");
    RTX_OPTION("rtx.restirFG", float, causticSpatialRadius, 3.0f, "Pixel radius for caustic spatial resampling. Very small radius (1-2 pixels) recommended.");

    // Bias Correction
    RTX_OPTION("rtx.restirFG", ReSTIRFGBiasCorrectionMode, biasCorrectionMode, ReSTIRFGBiasCorrectionMode::Raytraced, "Bias correction mode for reservoir resampling.");
    RTX_OPTION("rtx.restirFG", float, pairwiseMISCentralWeight, 0.1f, "Central weight for pairwise MIS.");
    RTX_OPTION("rtx.restirFG", float, maxLuminance, 25.0f, "Maximum luminance for firefly suppression.");
    RTX_OPTION("rtx.restirFG", float, minPhotonContribution, 0.002f, "Minimum photon contribution threshold.");

    // Performance Options
    RTX_OPTION("rtx.restirFG", bool, usePhotonCulling, true, "Use photon culling to reduce acceleration structure build time.");
    RTX_OPTION("rtx.restirFG", bool, useSplitCollection, false, "Split photon collection into separate FG and caustic passes.");
    RTX_OPTION("rtx.restirFG", bool, useStochasticCollection, true, "Use stochastic photon collection for variance reduction.");

    // --- Resources ---

    // Photon Buffers
    Rc<DxvkBuffer> m_photonBuffer;                    // Photon data (position, flux, direction)
    Rc<DxvkBuffer> m_photonAABBBuffer[2];             // AABB data for global [0] and caustic [1] photons
    Rc<DxvkBuffer> m_photonCounterBuffer;             // Atomic counter for photon emission
    
    // Photon Acceleration Structure
    Rc<DxvkBuffer> m_photonASBuffer;                  // Buffer backing the acceleration structure
    Rc<DxvkAccelStructure> m_photonAccelStructure;    // Photon BLAS for efficient lookup
    Rc<DxvkBuffer> m_photonScratchBuffer;             // Scratch buffer for AS builds
    Resources::Resource m_photonAS;                   // Legacy resource reference

    // Reservoir Buffers (double-buffered for temporal reuse)
    Rc<DxvkBuffer> m_fgReservoirBuffer;               // Final Gather reservoirs
    Rc<DxvkBuffer> m_causticReservoirBuffer;          // Caustic photon reservoirs
    Rc<DxvkBuffer> m_fgSampleBuffer;                  // Final Gather sample data
    Rc<DxvkBuffer> m_causticSampleBuffer;             // Caustic sample data

    // Surface Buffers
    Rc<DxvkBuffer> m_surfaceBuffer;                   // Surface data for resampling

    // Output Textures
    Resources::Resource m_fgRadiance;                 // Final Gather radiance output
    Resources::Resource m_causticRadiance;            // Caustic radiance output
    Resources::AliasedResource m_combinedRadiance;    // Combined output (aliased with composite)

    // Frame tracking
    uint32_t m_frameCount = 0;
    bool m_canResample = false;  // True if previous frame data is valid
  };
}
