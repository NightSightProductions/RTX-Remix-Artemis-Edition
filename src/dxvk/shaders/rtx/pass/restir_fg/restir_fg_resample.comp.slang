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

// ReSTIR-FG Final Gather Reservoir Resampling Pass
// Performs spatiotemporal resampling of final gather reservoirs

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_types.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/sampling.slangh"
#include "rtx/utility/gpu_printing.slangh"
#include "rtx/concept/camera/camera.slangh"

// Inputs
layout(rg16f, binding = RESTIR_FG_RESAMPLE_BINDING_MVEC_INPUT)
Texture2D<float2> MotionVector;

layout(rgba32f, binding = RESTIR_FG_RESAMPLE_BINDING_WORLD_POSITION_INPUT)
Texture2D<float4> WorldPosition;

layout(r32ui, binding = RESTIR_FG_RESAMPLE_BINDING_WORLD_NORMAL_INPUT)
Texture2D<uint> WorldNormal;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_PREV)
StructuredBuffer<PackedReSTIRFG_Reservoir> FGReservoirPrev;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SAMPLE_PREV)
StructuredBuffer<PackedReSTIRFG_Sample> FGSamplePrev;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SURFACE_PREV)
StructuredBuffer<PackedReSTIRFG_SurfaceData> SurfaceDataPrev;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SURFACE_CURR)
StructuredBuffer<PackedReSTIRFG_SurfaceData> SurfaceDataCurr;

// Inputs/Outputs
layout(binding = RESTIR_FG_RESAMPLE_BINDING_RESERVOIR_CURR)
RWStructuredBuffer<PackedReSTIRFG_Reservoir> FGReservoirCurr;

layout(binding = RESTIR_FG_RESAMPLE_BINDING_SAMPLE_CURR)
RWStructuredBuffer<PackedReSTIRFG_Sample> FGSampleCurr;

// Decode shading normal from packed format
float3 decodeNormal(uint packed)
{
  return signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(packed));
}

// Check if two surfaces are similar enough for resampling
bool areSurfacesSimilar(ReSTIRFG_SurfaceData curr, ReSTIRFG_SurfaceData prev)
{
  float normalSimilarity = dot(curr.normal, prev.normal);
  if (normalSimilarity < cb.restirFGNormalThreshold) {
    return false;
  }

  float currDepth = abs(cameraWorldPosToLinearViewZ(cb.camera, curr.position));
  float prevDepth = abs(cameraWorldPosToLinearViewZ(cb.camera, prev.position));
  float relDepth = abs(currDepth - prevDepth) / max(currDepth, 1.0f);
  if (relDepth > cb.restirFGDepthThreshold) {
    return false;
  }

  return true;
}

float evaluateTargetAtSurface(ReSTIRFG_SurfaceData surface, ReSTIRFG_Sample sample)
{
  float3 toSample = sample.hitPosition - surface.position;
  float distSq = dot(toSample, toSample);
  if (distSq <= 1e-8f) {
    return 0.0f;
  }

  float invDist = rsqrt(distSq);
  float3 dir = toSample * invDist;
  float NdotL = max(0.0f, dot(surface.normal, dir));
  if (NdotL <= 0.0f) {
    return 0.0f;
  }

  float Li = calcBt709Luminance(sample.radiance);
  if (!isfinite(Li) || Li <= 0.0f) {
    return 0.0f;
  }

  // Keep resampling target bounded to avoid unstable reservoir updates from extreme outliers.
  float maxLuminance = cb.restirFGMaxLuminance;
  const bool isEmissiveSample = (sample.surfaceHit == 3u);
  float clampLuminance = isEmissiveSample ? (maxLuminance * 16.0f) : maxLuminance;
  Li = min(Li, max(clampLuminance, 1e-6f));

  return Li * NdotL;
}

bool reprojectToPreviousPixel(uint2 pixel, ReSTIRFG_SurfaceData currentSurface, uint2 resolution, out float2 prevPixelF)
{
  float2 motionPrevPixel = MotionVector[pixel].xy + float2(pixel);
  bool motionValid = all(motionPrevPixel >= float2(0.0f)) && all(motionPrevPixel < float2(resolution));

  // Prefer world-space reprojection when possible for more stable temporal matching.
  float4 prevNdc = mul(cb.camera.prevWorldToProjection, float4(currentSurface.position, 1.0f));
  if (abs(prevNdc.w) > 1e-6f) {
    prevNdc.xyz /= prevNdc.w;

    bool inClip = all(prevNdc.xyz <= float3(1.0f, 1.0f, 1.0f)) && all(prevNdc.xyz >= float3(-1.0f, -1.0f, 0.0f));
    if (inClip) {
      float2 uv = cameraNDCToScreenUV(cb.camera, prevNdc.xy);
      prevPixelF = uv * float2(resolution);
      return all(prevPixelF >= float2(0.0f)) && all(prevPixelF < float2(resolution));
    }
  }

  prevPixelF = motionPrevPixel;
  return motionValid;
}

// Combine two reservoirs using RIS
bool combineReservoirs(
  inout ReSTIRFG_Reservoir combined,
  inout ReSTIRFG_Sample combinedSample,
  ReSTIRFG_Reservoir candidate,
  ReSTIRFG_Sample candidateSample,
  float targetAtCombined,
  inout RNG rng)
{
  if (candidate.M == 0 || candidate.wSum <= 0.0f) {
    return false;
  }
  
  float candidateWeight = targetAtCombined * candidate.wSum * float(candidate.M);
  if (!isfinite(candidateWeight) || candidateWeight <= 0.0f) {
    return false;
  }
  
  combined.wSum += candidateWeight;
  combined.M += candidate.M;
  
  // Reservoir selection
  float rand = getNextSampleBlueNoise(rng);
  if (rand * combined.wSum < candidateWeight) {
    combined.targetFunc = targetAtCombined;
    combinedSample = candidateSample;
    return true;
  }
  
  return false;
}

uint getTemporalHistoryLimit(uint baseLimit, bool spatialEnabled, ReSTIRFG_Reservoir currentReservoir)
{
  // TU uses a longer cap for pure temporal mode (scaled by current confidence).
  if (!spatialEnabled) {
    return baseLimit * max(1u, currentReservoir.M);
  }

  return baseLimit;
}

static const uint kBiasCacheMaxSamples = 8u;
static const float kMaxNormalizedReservoirWeight = 64.0f;

bool isOccludedForBias(float3 surfacePos, float3 surfaceNormal, float3 targetPos)
{
  float3 toTarget = targetPos - surfacePos;
  float dist = length(toTarget);
  if (dist <= 1e-4f) {
    return false;
  }

  float3 dir = toTarget / dist;

  RayDesc rayDesc;
  rayDesc.Origin = surfacePos + surfaceNormal * 1e-3f;
  rayDesc.Direction = dir;
  rayDesc.TMin = 0.0f;
  rayDesc.TMax = max(0.0f, dist * 0.999f);

  const uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;
  RayQuery<rayFlags> q;
  q.TraceRayInline(topLevelAS, rayFlags, 0xFF, rayDesc);
  while (q.Proceed()) {
  }

  return q.CommittedStatus() != COMMITTED_NOTHING;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint2 pixel = dispatchThreadId.xy;
  uint width, height;
  MotionVector.GetDimensions(width, height);
  const uint2 resolution = uint2(width, height);
  
  if (any(pixel >= resolution)) {
    return;
  }
  
  uint linearIdx = pixel.y * resolution.x + pixel.x;
  
  // Initialize RNG
  RNG rng = createRNG(uvec2(pixel), cb.frameIdx);
  
  const uint kResamplingModeTemporal = 0u;
  const uint kResamplingModeSpatial = 1u;

  // Load current reservoir and surface.
  // TU parity: spatial mode starts from previous-buffer reservoirs.
  const bool modeSpatial = (cb.restirFGResamplingMode == kResamplingModeSpatial);
  ReSTIRFG_Reservoir currentReservoir = modeSpatial ? FGReservoirPrev[linearIdx].unpack() : FGReservoirCurr[linearIdx].unpack();
  ReSTIRFG_Sample currentSample = modeSpatial ? FGSamplePrev[linearIdx].unpack() : FGSampleCurr[linearIdx].unpack();
  ReSTIRFG_SurfaceData currentSurface = SurfaceDataCurr[linearIdx].unpack();
  
  // Skip invalid surfaces
  if (currentSurface.hitInfo == 0) {
    return;
  }
  
  // Initialize using the same target-space evaluation as all reused candidates.
  ReSTIRFG_Reservoir combinedReservoir = { 0.0f, 0.0f, 0u, 0u };
  ReSTIRFG_Sample combinedSample = currentSample;
  float currentTargetAtCurrent = evaluateTargetAtSurface(currentSurface, currentSample);
  if (isfinite(currentTargetAtCurrent) && currentTargetAtCurrent > 0.0f) {
    currentReservoir.targetFunc = currentTargetAtCurrent;
  }
  combineReservoirs(combinedReservoir, combinedSample, currentReservoir, currentSample, currentTargetAtCurrent, rng);
  
  const bool enableTemporal = cb.restirFGResamplingMode != kResamplingModeSpatial;
  const bool enableSpatial = cb.restirFGResamplingMode != kResamplingModeTemporal;
  const uint maxHistory = getTemporalHistoryLimit(cb.restirFGTemporalHistoryLength, enableSpatial, currentReservoir);
  bool foundTemporalSurface = false;
  bool usedTemporalReservoir = false;
  float2 prevPixelF = float2(pixel);
  int2 prevPixelBase = int2(pixel);
  int2 selectedPrevPixel = int2(pixel);
  ReSTIRFG_SurfaceData selectedTemporalSurface = currentSurface;
  bool hasReprojectedPixel = false;
  uint temporalReservoirAge = 0u;
  uint selectedCandidateIndex = 0xFFFFFFFFu;
  uint cachedCount = 0u;
  ReSTIRFG_SurfaceData cachedSurfaces[kBiasCacheMaxSamples];
  uint cachedM[kBiasCacheMaxSamples];
  float debugTemporalTargetAtCurrent = -1.0f;

  if (enableTemporal) {
    if (reprojectToPreviousPixel(pixel, currentSurface, resolution, prevPixelF)) {
      prevPixelBase = int2(round(prevPixelF));

      hasReprojectedPixel = true;
      selectedPrevPixel = prevPixelBase;
      const float temporalSearchRadius = 4.0f;
      for (uint i = 0; i < 9; i++) {
        int2 offset = int2(0);
        if (i > 0) {
          float2 randomOffset = float2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng)) - 0.5f;
          offset = int2(randomOffset * temporalSearchRadius);
        }

        int2 candidatePixel = prevPixelBase + offset;
        if (any(candidatePixel < int2(0)) || any(candidatePixel >= int2(resolution))) {
          continue;
        }

        uint candidateLinearIdx = candidatePixel.y * resolution.x + candidatePixel.x;
        ReSTIRFG_SurfaceData candidateSurface = SurfaceDataPrev[candidateLinearIdx].unpack();
        if (candidateSurface.hitInfo == 0) {
          continue;
        }

        if (!areSurfacesSimilar(currentSurface, candidateSurface)) {
          continue;
        }

        selectedTemporalSurface = candidateSurface;
        selectedPrevPixel = candidatePixel;
        foundTemporalSurface = true;
        break;
      }

      if (foundTemporalSurface) {
        uint prevLinearIdx = selectedPrevPixel.y * resolution.x + selectedPrevPixel.x;

        ReSTIRFG_Reservoir prevReservoir = FGReservoirPrev[prevLinearIdx].unpack();
        ReSTIRFG_Sample prevSample = FGSamplePrev[prevLinearIdx].unpack();
        temporalReservoirAge = prevReservoir.age;

        if (prevReservoir.M > maxHistory) {
          prevReservoir.M = maxHistory;
        }

        if (prevReservoir.M > 0u && cachedCount < kBiasCacheMaxSamples) {
          cachedSurfaces[cachedCount] = selectedTemporalSurface;
          cachedM[cachedCount] = prevReservoir.M;
        }

        float targetAtCurrent = evaluateTargetAtSurface(currentSurface, prevSample);
        debugTemporalTargetAtCurrent = targetAtCurrent;
        usedTemporalReservoir = combineReservoirs(combinedReservoir, combinedSample, prevReservoir, prevSample, targetAtCurrent, rng);
        if (usedTemporalReservoir && cachedCount < kBiasCacheMaxSamples) {
          selectedCandidateIndex = cachedCount;
        }
        if (prevReservoir.M > 0u && cachedCount < kBiasCacheMaxSamples) {
          cachedCount++;
        }
      }
    }
  }

  if (enableSpatial) {
    // Spatial resampling from previous-frame neighbors around reprojected center.
    uint spatialSamples = modeSpatial
      ? cb.restirFGSpatialSamples
      : (foundTemporalSurface ? cb.restirFGSpatialSamples : (cb.restirFGSpatialSamples + cb.restirFGDisocclusionBoostSamples));

    float spatialRadius = cb.restirFGSpatialRadius;
    float2 spatialCenter = modeSpatial ? float2(pixel) : (hasReprojectedPixel ? prevPixelF : float2(pixel));

    for (uint i = 0; i < spatialSamples; i++) {
      float2 randSamp = (float2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng)) * 2.0f) - 1.0f;
      int2 spatialOffset = int2(round(randSamp * spatialRadius));
      int2 neighborPixel = int2(round(spatialCenter)) + spatialOffset;

      if (foundTemporalSurface && all(neighborPixel == selectedPrevPixel)) {
        continue;
      }

      if (all(neighborPixel >= int2(0)) && all(neighborPixel < int2(resolution))) {
        uint neighborLinearIdx = neighborPixel.y * resolution.x + neighborPixel.x;

        ReSTIRFG_Reservoir neighborReservoir = FGReservoirPrev[neighborLinearIdx].unpack();
        ReSTIRFG_Sample neighborSample = FGSamplePrev[neighborLinearIdx].unpack();
        ReSTIRFG_SurfaceData neighborSurface = SurfaceDataPrev[neighborLinearIdx].unpack();

        if (!modeSpatial && neighborReservoir.M > maxHistory) {
          neighborReservoir.M = maxHistory;
        }

        // Check surface similarity
        ReSTIRFG_SurfaceData compareSurface = modeSpatial ? SurfaceDataCurr[neighborLinearIdx].unpack() : neighborSurface;
        if (areSurfacesSimilar(currentSurface, compareSurface)) {
          if (neighborReservoir.M > 0u && cachedCount < kBiasCacheMaxSamples) {
            cachedSurfaces[cachedCount] = compareSurface;
            cachedM[cachedCount] = neighborReservoir.M;
          }

          // Re-evaluate target function at current surface
          float targetAtCurrent = evaluateTargetAtSurface(currentSurface, neighborSample);

          // Combine with spatial neighbor
          bool selected = combineReservoirs(combinedReservoir, combinedSample, neighborReservoir, neighborSample, targetAtCurrent, rng);
          if (selected && neighborReservoir.M > 0u && cachedCount < kBiasCacheMaxSamples) {
            selectedCandidateIndex = cachedCount;
          }
          if (neighborReservoir.M > 0u && cachedCount < kBiasCacheMaxSamples) {
            cachedCount++;
          }
        }
      }
    }
  }
  
  // Normalize final weight
  if (combinedReservoir.M > 0 && combinedReservoir.targetFunc > 0) {
    if (cb.restirFGBiasCorrectionMode >= 1u) {
      // TU-style neighbor-aware MIS finalization for bias-corrected modes.
      float pi = combinedReservoir.targetFunc;
      float piSum = combinedReservoir.targetFunc;

      for (uint i = 0u; i < cachedCount; i++) {
        float ps = evaluateTargetAtSurface(cachedSurfaces[i], combinedSample);

        if (cb.restirFGBiasCorrectionMode == 2u && ps > 0.0f) {
          if (isOccludedForBias(cachedSurfaces[i].position, cachedSurfaces[i].normal, combinedSample.hitPosition)) {
            ps = 0.0f;
          }
        }

        if (selectedCandidateIndex == i) {
          pi = ps;
        }
        piSum += ps * float(cachedM[i]);
      }

      float denom = combinedReservoir.targetFunc * max(piSum, 1e-8f);
      combinedReservoir.wSum = (denom > 0.0f) ? (combinedReservoir.wSum * pi / denom) : 0.0f;
    } else {
      combinedReservoir.wSum = combinedReservoir.wSum / (combinedReservoir.targetFunc * float(combinedReservoir.M));
    }

    if (!isfinite(combinedReservoir.wSum) || combinedReservoir.wSum < 0.0f) {
      combinedReservoir.wSum = 0.0f;
      combinedReservoir.targetFunc = 0.0f;
      combinedReservoir.M = 0;
    } else {
      combinedReservoir.wSum = min(combinedReservoir.wSum, kMaxNormalizedReservoirWeight);
    }
  } else {
    combinedReservoir.wSum = 0.0f;
    combinedReservoir.targetFunc = 0.0f;
    combinedReservoir.M = 0;
  }
  
  // Age tracks true temporal reuse and is reset when no valid temporal sample is used.
  combinedReservoir.age = usedTemporalReservoir ? (temporalReservoirAge + 1u) : 0u;

  // GPU print payload for ReSTIR-FG resampling diagnostics.
  // x: final wSum, y: final target, z: final M,
  // w: bit flags (1 reprojected, 2 temporal surface, 4 temporal used, 8 spatial enabled, 16 temporal enabled)
  //    plus a small frame tag to avoid duplicate suppression in CPU-side logging.
  float debugFlags = float((hasReprojectedPixel ? 1u : 0u)
                         | (foundTemporalSurface ? 2u : 0u)
                         | (usedTemporalReservoir ? 4u : 0u)
                         | (enableSpatial ? 8u : 0u)
                         | (enableTemporal ? 16u : 0u));

  float debugFrameTag = float(cb.frameIdx % 1000u) * 0.001f;
  float4 debugData = float4(combinedReservoir.wSum, combinedReservoir.targetFunc, float(combinedReservoir.M), debugFlags + debugFrameTag);
  GpuPrint::write(u16vec2(pixel), debugData);
  
  // Write back results
  FGReservoirCurr[linearIdx] = PackedReSTIRFG_Reservoir::pack(combinedReservoir);
  FGSampleCurr[linearIdx] = PackedReSTIRFG_Sample::pack(combinedSample);
}
