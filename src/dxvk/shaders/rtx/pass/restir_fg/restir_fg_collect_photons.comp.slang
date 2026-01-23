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

// ReSTIR-FG Photon Collection Pass
// Traces final gather rays from surfaces and collects photons to initialize reservoirs
// Implements: "Generate Initial Final Gather and Caustic Reservoirs" from the algorithm

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_types.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/sampling.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/utility/geometry_flags.slangh"
#include "rtx/utility/gbuffer_helpers.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/concept/ray/ray.slangh"
#include "rtx/concept/ray/ray_helper.slangh"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/concept/surface/surface.slangh"
#include "rtx/concept/surface_material/surface_material.slangh"
#include "rtx/concept/light/light.slangh"

// GBuffer Inputs (matching ReSTIR GI pattern)
layout(r16ui, binding = RESTIR_FG_COLLECT_BINDING_SHARED_FLAGS_INPUT)
Texture2D<uint16_t> SharedFlags;

layout(r16ui, binding = RESTIR_FG_COLLECT_BINDING_SHARED_SURFACE_INDEX_INPUT)
Texture2D<uint16_t> SharedSurfaceIndex;

layout(r32ui, binding = RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT)
Texture2D<uint> PrimaryWorldShadingNormal;

layout(r8, binding = RESTIR_FG_COLLECT_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT)
Texture2D<float> PrimaryPerceptualRoughness;

layout(rg16_snorm, binding = RESTIR_FG_COLLECT_BINDING_PRIMARY_VIEW_DIRECTION_INPUT)
Texture2D<float2> PrimaryViewDirection;

layout(r16f, binding = RESTIR_FG_COLLECT_BINDING_PRIMARY_CONE_RADIUS_INPUT)
Texture2D<float> PrimaryConeRadius;

layout(rgba32f, binding = RESTIR_FG_COLLECT_BINDING_PRIMARY_WORLD_POSITION_INPUT)
Texture2D<float4> PrimaryWorldPositionWorldTriangleNormal;

layout(r32f, binding = RESTIR_FG_COLLECT_BINDING_PRIMARY_POSITION_ERROR_INPUT)
Texture2D<float> PrimaryPositionError;

// Photon data
layout(binding = RESTIR_FG_COLLECT_BINDING_PHOTON_AS)
RaytracingAccelerationStructure PhotonAS;

layout(binding = RESTIR_FG_COLLECT_BINDING_PHOTON_DATA)
StructuredBuffer<PackedPhotonData> PhotonBuffer;

layout(binding = RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_GLOBAL)
StructuredBuffer<float> PhotonAABBGlobal;

layout(binding = RESTIR_FG_COLLECT_BINDING_PHOTON_AABB_CAUSTIC)
StructuredBuffer<float> PhotonAABBCaustic;

layout(binding = RESTIR_FG_COLLECT_BINDING_PHOTON_COUNTER)
StructuredBuffer<uint> PhotonCounter;

// Outputs
layout(binding = RESTIR_FG_COLLECT_BINDING_FG_RESERVOIR_OUTPUT)
RWStructuredBuffer<PackedReSTIRFG_Reservoir> FGReservoirOutput;

layout(binding = RESTIR_FG_COLLECT_BINDING_FG_SAMPLE_OUTPUT)
RWStructuredBuffer<PackedReSTIRFG_Sample> FGSampleOutput;

layout(binding = RESTIR_FG_COLLECT_BINDING_CAUSTIC_RESERVOIR_OUTPUT)
RWStructuredBuffer<PackedReSTIRFG_CausticReservoir> CausticReservoirOutput;

layout(binding = RESTIR_FG_COLLECT_BINDING_CAUSTIC_SAMPLE_OUTPUT)
RWStructuredBuffer<PackedReSTIRFG_CausticSample> CausticSampleOutput;

layout(binding = RESTIR_FG_COLLECT_BINDING_SURFACE_DATA_OUTPUT)
RWStructuredBuffer<PackedReSTIRFG_SurfaceData> SurfaceDataOutput; // Store surface info for temporal reuse

// Decode shading normal
float3 decodeNormal(uint packed)
{
  return signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(packed));
}

// Check if surface is diffuse based on roughness
bool isDiffuseSurface(float roughness)
{
  return roughness >= cb.restirFGRoughnessThreshold;
}

// Query photons near a position using rayQuery on the photon acceleration structure
void queryPhotonsWithRayQuery(float3 position, float3 normal, float searchRadius,
                               bool includeCaustic, inout float3 accumulatedRadiance, 
                               inout uint photonCount, inout float3 bestPhotonDir)
{
  // Use a small sphere ray cast to intersect with photon AABBs
  // The photon AS contains AABB geometries representing photon collection radii
  
  RayDesc rayDesc;
  rayDesc.Origin = position;
  // Cast a very short ray in multiple directions to find nearby photons
  // Alternatively, use the fact that photon AABBs intersect with our position
  
  // For AABB-based photon collection, we need to check if our position
  // is within any photon's AABB. We do this by casting a zero-length ray
  // or by using intersection queries.
  
  // Actually, the proper way is to cast a ray from our position and see
  // which photon AABBs we intersect. Since photons have radii, we check
  // if the ray origin is within any photon's sphere.
  
  // Cast rays in multiple random directions to find photons
  float3 directions[6] = {
    float3(1, 0, 0), float3(-1, 0, 0),
    float3(0, 1, 0), float3(0, -1, 0),
    float3(0, 0, 1), float3(0, 0, -1)
  };
  
  for (int d = 0; d < 6; d++) {
    rayDesc.Direction = directions[d];
    rayDesc.TMin = 0.0f;
    rayDesc.TMax = searchRadius;
    
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(PhotonAS, RAY_FLAG_NONE, 0xFF, rayDesc);
    
    while (q.Proceed()) {
      if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
        // Get the photon index from the primitive ID
        uint photonIdx = q.CandidatePrimitiveIndex();
        uint geometryIdx = q.CandidateGeometryIndex();
        
        // Geometry 0 = global, 1 = caustic
        bool isCausticPhoton = (geometryIdx == 1);
        
        // Skip caustic photons if not requested
        if (isCausticPhoton && !includeCaustic) {
          continue;
        }
        
        // Calculate buffer index
        uint bufferIndex = isCausticPhoton ? (cb.restirFGPhotonsPerFrame + photonIdx) : photonIdx;
        
        PhotonData photon = PhotonBuffer[bufferIndex].unpack();
        
        float3 toPhoton = photon.position - position;
        float distSq = dot(toPhoton, toPhoton);
        float photonRadiusSq = photon.radius * photon.radius;
        
        // Check if we're within the photon's collection radius
        if (distSq < photonRadiusSq) {
          float3 photonDir = photon.getDirection();
          float NdotL = dot(normal, -photonDir);
          
          if (NdotL > 0.0f) {
            // Epanechnikov kernel for smooth falloff
            float dist = sqrt(distSq);
            float kernel = 1.0f - (distSq / photonRadiusSq);
            kernel = max(0.0f, kernel);
            
            // Normalize by photon area (pi * r^2)
            float normalization = 1.0f / (pi * photonRadiusSq);
            
            accumulatedRadiance += photon.flux * kernel * normalization * NdotL;
            bestPhotonDir = photonDir;
            photonCount++;
          }
        }
        
        // Accept the hit to continue searching
        q.CommitProceduralPrimitiveHit(q.CandidateTriangleRayT());
      }
    }
  }
}

// Fallback: Query photons using direct buffer access (less efficient but simpler)
void queryPhotonsDirect(float3 position, float3 normal, float searchRadius,
                        bool includeGlobal, bool includeCaustic, inout float3 accumulatedRadiance,
                        uint globalPhotonCount, uint causticPhotonCount,
                        bool applyCosineTerm,
                        inout uint photonCount, inout float3 bestPhotonDir, inout float3 bestPhotonPos, inout RNG rng)
{
  uint maxPhotonsPerType = cb.restirFGPhotonsPerFrame;
  uint globalCount = min(globalPhotonCount, maxPhotonsPerType);
  uint causticCount = min(causticPhotonCount, maxPhotonsPerType);

  if ((!includeGlobal || globalCount == 0u) && (!includeCaustic || causticCount == 0u)) {
    return;
  }

  uint baseSampleCount = 64u;

  float searchRadiusSq = searchRadius * searchRadius;
  float causticRadiusSq = cb.restirFGCausticPhotonRadius * cb.restirFGCausticPhotonRadius;
  float bestContributionLuma = 0.0f;
  
  if (includeGlobal && globalCount > 0u) {
    uint sampleCount = min(baseSampleCount, globalCount);
    uint stride = max(1u, globalCount / max(sampleCount, 1u));
    float sampleNormalization = float(globalCount) / float(max(sampleCount, 1u));
    uint globalBaseOffset = uint(getNextSampleBlueNoise(rng) * float(globalCount));

    for (uint i = 0; i < sampleCount; i++) {
      uint photonIdx = (globalBaseOffset + i * stride) % globalCount;
      
      PhotonData photon = PhotonBuffer[photonIdx].unpack();
      
      float3 toPhoton = photon.position - position;
      float distSq = dot(toPhoton, toPhoton);
      
      if (distSq < searchRadiusSq && distSq > 0.0001f) {
        float3 photonDir = photon.getDirection();
        float NdotL = dot(normal, -photonDir);
        
        if (NdotL > 0.0f) {
          float dist = sqrt(distSq);
          // Epanechnikov kernel
          float kernel = 1.0f - (distSq / searchRadiusSq);
          kernel = max(0.0f, kernel);
          
          // Normalize by area
          float normalization = 1.0f / (pi * searchRadiusSq);

          float3 contribution = photon.flux * kernel * normalization * sampleNormalization;
          if (applyCosineTerm) {
            contribution *= NdotL;
          }
          accumulatedRadiance += contribution;

          float contributionLuma = calcBt709Luminance(contribution);
          if (contributionLuma > bestContributionLuma) {
            bestContributionLuma = contributionLuma;
            bestPhotonDir = photonDir;
            bestPhotonPos = photon.position;
          }

          photonCount++;
        }
      }
    }
  }
  
  // Also check caustic photons if requested
  if (includeCaustic && causticCount > 0u) {
    uint sampleCount = min(baseSampleCount, causticCount);
    uint stride = max(1u, causticCount / max(sampleCount, 1u));
    float sampleNormalization = float(causticCount) / float(max(sampleCount, 1u));
    uint causticBaseOffset = uint(getNextSampleBlueNoise(rng) * float(causticCount));

    for (uint i = 0; i < sampleCount; i++) {
      uint localPhotonIdx = (causticBaseOffset + i * stride) % causticCount;
      uint photonIdx = cb.restirFGPhotonsPerFrame + localPhotonIdx;
      
      PhotonData photon = PhotonBuffer[photonIdx].unpack();
      
      float3 toPhoton = photon.position - position;
      float distSq = dot(toPhoton, toPhoton);

      if (distSq < causticRadiusSq && distSq > 0.0001f) {
        float3 photonDir = photon.getDirection();
        float NdotL = dot(normal, -photonDir);
        
        if (NdotL > 0.0f) {
          float kernel = 1.0f - (distSq / causticRadiusSq);
          kernel = max(0.0f, kernel);
          float normalization = 1.0f / (pi * causticRadiusSq);

          float3 contribution = photon.flux * kernel * normalization * sampleNormalization;
          if (applyCosineTerm) {
            contribution *= NdotL;
          }
          accumulatedRadiance += contribution;

          float contributionLuma = calcBt709Luminance(contribution);
          if (contributionLuma > bestContributionLuma) {
            bestContributionLuma = contributionLuma;
            bestPhotonDir = photonDir;
            bestPhotonPos = photon.position;
          }

          photonCount++;
        }
      }
    }
  }
}

// RTXDI-style multi-light sampling at a hit point
// Uses reservoir-based importance sampling to efficiently sample from all lights
// Leverages existing memoryPolymorphicLightSampleArea() infrastructure for proper light handling
float3 sampleDirectLightRTXDI(float3 hitPos, float3 hitNormal, float3 diffuseAlbedo, inout RNG rng)
{
  if (cb.lightCount == 0) {
    return float3(0);
  }
  
  // Create minimal surface interaction for light sampling
  MinimalSurfaceInteraction surfaceInteraction;
  surfaceInteraction.position = hitPos;
  surfaceInteraction.triangleNormal = f16vec3(hitNormal);
  
  // RTXDI-style weighted reservoir sampling for direct light
  // Sample multiple lights and use RIS to select the best one
  const uint numLightSamples = min(cb.lightCount, 8u);
  
  float wSum = 0.0f;
  float3 selectedRadiance = float3(0);
  float3 selectedLightPos = float3(0);
  float selectedLightDist = 0.0f;
  
  for (uint i = 0; i < numLightSamples; i++) {
    // Sample a random light from the pool
    float rand1 = getNextSampleBlueNoise(rng);
    uint lightIdx = uint(rand1 * float(cb.lightCount)) % cb.lightCount;
    
    // Get random UV for light surface sampling
    f16vec2 lightUV = f16vec2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng));
    
    // Use the existing polymorphic light sampling infrastructure
    // This properly handles all light types (sphere, rect, disk, cylinder, distant)
    MemoryPolymorphicLight memoryLight = lights[lightIdx];
    LightSample lightSample = memoryPolymorphicLightSampleArea(memoryLight, lightUV, surfaceInteraction);
    
    // Compute direction to light
    float3 toLight = lightSample.position - hitPos;
    float lightDist = length(toLight);
    float3 lightDir = lightDist > 0.001f ? toLight / lightDist : float3(0, 1, 0);
    
    // Check if light is on correct side of surface
    float NdotL = dot(hitNormal, lightDir);
    if (NdotL <= 0.0f || lightSample.solidAnglePdf <= 0.0f) {
      continue;
    }
    
    // Calculate target PDF (BRDF * Le / pdf) with material-aware diffuse response.
    float3 brdf = diffuseAlbedo / pi;
    float3 contribution = lightSample.radiance * brdf * NdotL / max(lightSample.solidAnglePdf, 1e-6f);
    float targetPdf = calcBt709Luminance(contribution);
    
    // Source PDF: uniform selection from light pool
    float sourcePdf = 1.0f / float(cb.lightCount);
    
    // RIS weight: targetPdf / sourcePdf
    float weight = targetPdf / sourcePdf;
    wSum += weight;
    
    // Reservoir sampling: select with probability weight/wSum
    float randSelect = getNextSampleBlueNoise(rng);
    if (randSelect * wSum < weight) {
      selectedRadiance = contribution;
      selectedLightPos = lightSample.position;
      selectedLightDist = lightDist;
    }
  }
  
  // If no valid light was selected, return zero
  if (wSum <= 0.0f || selectedLightDist <= 0.0f) {
    return float3(0);
  }
  
  // Check visibility of selected light with shadow ray
  float3 selectedLightDir = normalize(selectedLightPos - hitPos);
  
  RayDesc shadowRay;
  shadowRay.Origin = hitPos;
  shadowRay.Direction = selectedLightDir;
  shadowRay.TMin = 0.001f;
  shadowRay.TMax = selectedLightDist - 0.01f;
  
  RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
  shadowQuery.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 
                              OBJECT_MASK_OPAQUE, shadowRay);
  while (shadowQuery.Proceed()) {}
  
  if (shadowQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
    // Light is occluded
    return float3(0);
  }
  
  // Return weighted contribution (RIS weight normalization)
  // W = (1/M) * wSum / targetPdf_selected
  float targetPdfSelected = calcBt709Luminance(selectedRadiance);
  if (targetPdfSelected <= 0.0f) {
    return float3(0);
  }
  
  float W = wSum / (float(numLightSamples) * targetPdfSelected);
  return selectedRadiance * W;
}

// Trace a final gather ray and collect photons along the path
// If an FG ray hits a specular surface, stop and store 0-weight reservoir
// since these paths are already covered by caustic photon paths.
// Returns: 0 = miss, 1 = diffuse hit with photons, 2 = specular hit (store 0-weight)
int traceFinalGatherRay(float3 origin, float3 direction, f16vec3 viewDir, float3 throughput,
                        uint globalPhotonCount, uint causticPhotonCount,
                        inout ReSTIRFG_Sample outSample, inout RNG rng)
{
  const float kMinEmissiveLuminance = 1e-4f;
  float3 rayOrigin = origin;
  float3 rayDirection = normalize(direction);
  float3 currentThroughput = throughput;
  const uint maxBounces = max(1u, cb.restirFGMaxFinalGatherBounces);

  for (uint bounce = 0; bounce < maxBounces; ++bounce) {
    RayDesc rayDesc;
    rayDesc.Origin = rayOrigin;
    rayDesc.Direction = rayDirection;
    rayDesc.TMin = 0.001f;
    rayDesc.TMax = 10000.0f;

    RayQuery<0> q;
    q.TraceRayInline(
      topLevelAS,
      0,
      OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT,
      rayDesc);

    while (q.Proceed()) {}

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
      return 0;
    }

    RAY_HIT_INFO_CREATE_RAY_QUERY_COMMITTED(rayHitInfo, q)

    Ray ray;
    ray.origin = rayOrigin;
    ray.direction = rayDirection;
    ray.coneRadius = float16_t(0.0f);
    ray.spreadAngle = float16_t(cb.screenSpacePixelSpreadHalfAngle);

    RayInteraction rayInteraction = rayInteractionCreate(ray, rayHitInfo);

    float hitT = rayHitInfo.hitDistance;
    float3 hitPos = rayOrigin + rayDirection * hitT;

    uint surfaceIdx = calculateSurfaceIndex(rayHitInfo);
    Surface surface = surfaces[surfaceIdx];

    SurfaceInteraction surfaceInteraction = surfaceInteractionCreate<SurfaceIgnoreTangents>(
      surface, rayInteraction, ray, false, kFootprintFromRayDirection);

    float3 hitNormal = float3(surfaceInteraction.triangleNormal);

    const MemoryPolymorphicSurfaceMaterial memMaterial = surfaceMaterials[surfaceIdx];
    const uint8_t materialType = rayInteraction.materialType;

    float hitRoughness = 0.5f;
    float3 diffuseAlbedo = float3(0.8f);
    bool isTranslucentSpecular = false;
    float translucentIor = 1.0f;
    float3 emissiveRadiance = float3(0);
    if (materialType == surfaceMaterialTypeOpaque) {
      const OpaqueSurfaceMaterial opaqueMaterial = opaqueSurfaceMaterialCreate(memMaterial, true);
      const OpaqueSurfaceMaterialInteraction materialInteraction =
        opaqueSurfaceMaterialInteractionCreate(opaqueMaterial, surface, surfaceInteraction, rayInteraction);
      hitRoughness = float(materialInteraction.isotropicRoughness);
      diffuseAlbedo = saturate(float3(materialInteraction.albedo));
      emissiveRadiance = float3(materialInteraction.emissiveRadiance);
    } else if (materialType == surfaceMaterialTypeTranslucent) {
      const TranslucentSurfaceMaterial translucentMaterial = translucentSurfaceMaterialCreate(memMaterial, true);
      const TranslucentSurfaceMaterialInteraction translucentInteraction =
        translucentSurfaceMaterialInteractionCreate(translucentMaterial, surface, surfaceInteraction, rayInteraction);
      // Treat translucent interfaces as specular transport for FG path continuation.
      hitRoughness = 0.0f;
      isTranslucentSpecular = true;
      translucentIor = max(float(translucentInteraction.refractiveIndex), 1.001f);
      emissiveRadiance = float3(translucentInteraction.emissiveRadiance);
    }

    float emissiveLuminance = calcBt709Luminance(emissiveRadiance);

    if (!isDiffuseSurface(hitRoughness)) {
      if (emissiveLuminance > kMinEmissiveLuminance) {
        outSample.hitPosition = hitPos;
        outSample.hitNormal = hitNormal;
        outSample.radiance = emissiveRadiance * currentThroughput;
        outSample.throughput = currentThroughput;
        outSample.pdf = 1.0f / (2.0f * pi);
        outSample.pathLength = bounce + 1;
        outSample.surfaceHit = 3;
        return 1;
      }

      if (bounce + 1 >= maxBounces) {
        outSample.hitPosition = hitPos;
        outSample.hitNormal = hitNormal;
        outSample.radiance = float3(0);
        outSample.throughput = float3(0);
        outSample.pdf = 1.0f;
        outSample.pathLength = bounce + 1;
        outSample.surfaceHit = 2;
        return 2;
      }

      // Avoid fixed per-bounce attenuation here; it over-darkens multi-bounce FG paths.

      float3 nextDir = reflect(rayDirection, hitNormal);
      if (isTranslucentSpecular) {
        float3 transportNormal = hitNormal;
        float etaI = 1.0f;
        float etaT = translucentIor;

        if (dot(rayDirection, transportNormal) > 0.0f) {
          transportNormal = -transportNormal;
          float tmpEta = etaI;
          etaI = etaT;
          etaT = tmpEta;
        }

        float eta = etaI / max(etaT, 1e-4f);
        float3 refracted = refract(rayDirection, transportNormal, eta);
        if (dot(refracted, refracted) > 1e-8f) {
          nextDir = refracted;
        }
      }

      rayOrigin = hitPos + hitNormal * 0.001f;
      rayDirection = normalize(nextDir);
      continue;
    }

    float3 photonRadiance = float3(0);
    uint photonCount = 0;
    float3 bestPhotonDir = float3(0, -1, 0);
    float3 bestPhotonPos = hitPos;

        queryPhotonsDirect(hitPos, hitNormal, cb.restirFGPhotonRadius, true, false,
          photonRadiance, globalPhotonCount, causticPhotonCount, true,
          photonCount, bestPhotonDir, bestPhotonPos, rng);

    float3 diffuseBounceRadiance = photonRadiance * diffuseAlbedo / pi;
    float3 directLight = sampleDirectLightRTXDI(hitPos, hitNormal, diffuseAlbedo, rng);
    float3 totalRadiance = (diffuseBounceRadiance + directLight + emissiveRadiance) * currentThroughput;

    if (photonCount > 0 || any(directLight > float3(0)) || any(emissiveRadiance > float3(0))) {
      outSample.hitPosition = hitPos;
      outSample.hitNormal = hitNormal;
      outSample.radiance = totalRadiance;
      outSample.throughput = currentThroughput;
      outSample.pdf = 1.0f / (2.0f * pi);
      outSample.pathLength = bounce + 1;
      outSample.surfaceHit = (emissiveLuminance > kMinEmissiveLuminance) ? 3 : 1;
      return 1;
    }

    return 0;
  }

  return 0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint2 pixel = dispatchThreadId.xy;
  uint width, height;
  SharedFlags.GetDimensions(width, height);
  const uint2 resolution = uint2(width, height);
  
  if (any(pixel >= resolution)) {
    return;
  }
  
  uint linearIdx = pixel.y * resolution.x + pixel.x;
  
  // Read surface data from GBuffer (like ReSTIR GI does)
  const Ray originalPrimaryRay = rayCreatePrimaryFromPixel(cb.camera, pixel);
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(pixel, SharedFlags);
  const bool gBufferMiss = isGBufferMiss(pixel, PrimaryConeRadius);
  
  // Check for valid surface
  bool isValidSurface = !gBufferMiss;
  
  float3 surfacePos, surfaceNormal, viewDir;
  float roughness;
  
  if (isValidSurface) {
    // Read full surface interaction data from GBuffer
    MinimalRayInteraction minimalRayInteraction = minimalRayInteractionReadFromGBuffer(
      originalPrimaryRay, pixel, PrimaryViewDirection, PrimaryConeRadius);
    
    MinimalSurfaceInteraction minimalSurfaceInteraction = minimalSurfaceInteractionReadFromGBufferWithError(
      pixel, PrimaryWorldPositionWorldTriangleNormal, PrimaryPositionError);
    
    // Read material properties
    f16vec3 shadingNormal = signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(PrimaryWorldShadingNormal[pixel]));
    roughness = PrimaryPerceptualRoughness[pixel];
    
    surfacePos = minimalSurfaceInteraction.position;
    surfaceNormal = float3(shadingNormal);
    viewDir = float3(minimalRayInteraction.viewDirection);
  } else {
    surfacePos = float3(0);
    surfaceNormal = float3(0, 1, 0);
    viewDir = float3(0, 0, -1);
    roughness = 1.0f;
  }
  
  // Initialize RNG
  RNG rng = createRNG(uvec2(pixel), cb.frameIdx);
  uint globalPhotonCount = min(PhotonCounter[0], cb.restirFGPhotonsPerFrame);
  uint causticPhotonCount = min(PhotonCounter[1], cb.restirFGPhotonsPerFrame);
  
  // Initialize output reservoirs and samples
  ReSTIRFG_Reservoir fgReservoir;
  fgReservoir.wSum = 0.0f;
  fgReservoir.targetFunc = 0.0f;
  fgReservoir.M = 0;
  fgReservoir.age = 0;
  
  ReSTIRFG_Sample fgSample;
  fgSample.hitPosition = float3(0);
  fgSample.hitNormal = float3(0, 1, 0);
  fgSample.radiance = float3(0);
  fgSample.throughput = float3(1);
  fgSample.pdf = 0.0f;
  fgSample.pathLength = 0;
  fgSample.surfaceHit = 0;
  
  ReSTIRFG_CausticReservoir causticReservoir;
  causticReservoir.wSum = 0.0f;
  causticReservoir.targetFunc = 0.0f;
  causticReservoir.M = 0;
  causticReservoir.pathLength = 0;
  
  ReSTIRFG_CausticSample causticSample;
  causticSample.photonPosition = float3(0);
  causticSample.photonDirection = float3(0, -1, 0);
  causticSample.photonFlux = float3(0);
  
  if (isValidSurface) {
    bool isDiffuse = isDiffuseSurface(roughness);
    float3 finalGatherOrigin = surfacePos;
    float3 finalGatherNormal = surfaceNormal;
    bool primaryIsSpecular = !isDiffuse;
    
    // If surface is not diffuse, trace until we find a diffuse surface
    // (Algorithm requirement: "If VBuffer surface is not diffuse, trace until one is found")
    if (!isDiffuse) {
      // Trace through specular surfaces to find diffuse surface
      float3 reflectDir = reflect(-viewDir, finalGatherNormal);
      
      // Trace specular chain to find diffuse surface
      RayDesc specRayDesc;
      specRayDesc.Origin = surfacePos;
      specRayDesc.Direction = reflectDir;
      specRayDesc.TMin = 0.001f;
      specRayDesc.TMax = 10000.0f;
      
      RayQuery<0> specQuery;
      specQuery.TraceRayInline(topLevelAS, 0, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, specRayDesc);
      
      while (specQuery.Proceed()) {}
      
      if (specQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float specT = specQuery.CommittedRayT();
        finalGatherOrigin = surfacePos + reflectDir * specT;
        // Approximate normal at new position
        finalGatherNormal = -reflectDir; // Temporary approximation
      }
    }
    
    // Algorithm Step: "For Final Gather, an additional ray is traced"
    // Trace a final gather ray from the (potentially refracted) diffuse surface
    float2 xi = float2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng));
    
    // Sample cosine-weighted hemisphere
    float solidAnglePdf;
    f16vec3 localDir = calcCosineHemisphereDirectionSample(f16vec2(xi), solidAnglePdf);
    
    // Transform to world space (build TBN from normal)
    float3 tangent, bitangent;
    if (abs(finalGatherNormal.y) > 0.999f) {
      tangent = float3(1, 0, 0);
    } else {
      tangent = normalize(cross(float3(0, 1, 0), finalGatherNormal));
    }
    bitangent = cross(finalGatherNormal, tangent);
    float3 fgDirection = normalize(tangent * float(localDir.x) + finalGatherNormal * float(localDir.y) + bitangent * float(localDir.z));
    
    int fgResult = traceFinalGatherRay(finalGatherOrigin, fgDirection, f16vec3(viewDir), float3(1),
                       globalPhotonCount, causticPhotonCount,
                       fgSample, rng);
    
    if (fgResult == 1) {
      // Diffuse hit with valid photon contribution
      float luminance = calcBt709Luminance(fgSample.radiance);
      
      if (luminance > 0.0f && float(solidAnglePdf) > 0.0f) {
        // Initialize with GI-style normalized weight (1 / selection PDF).
        // ReSTIR combine expects stored wSum to represent W, not target-scaled weight.
        fgReservoir.targetFunc = luminance;
        fgReservoir.wSum = 1.0f / max(float(solidAnglePdf), 1e-6f);
        fgReservoir.M = 1;
        fgSample.pdf = float(solidAnglePdf);
      }
    } else if (fgResult == 2) {
      // Specular surface hit - store 0-weight reservoir (covered by caustic photon paths)
      fgReservoir.wSum = 0.0f;
      fgReservoir.targetFunc = 0.0f;
      fgReservoir.M = 1; // Still count as having seen a sample
      fgReservoir.age = 0;
    }
    // fgResult == 0: miss, keep default 0-weight reservoir
    
    // Collect caustic photons at the primary surface
    // If first surface is non-diffuse, caustic reservoirs also include global photons
    float3 causticRadiance = float3(0);
    uint causticCount = 0;
    float3 causticPhotonDir = float3(0, -1, 0);
    float3 causticPhotonPos = surfacePos;
    float causticRadius = cb.restirFGCausticPhotonRadius;
    
    // For caustic reservoirs, keep diffuse surfaces caustic-only to avoid double-counting
    // with FG global-photon reuse. Allow global support when the primary hit is specular.
    bool includeGlobalInCaustic = primaryIsSpecular;
    bool includeCaustic = true;
    queryPhotonsDirect(surfacePos, surfaceNormal, causticRadius, includeGlobalInCaustic, includeCaustic,
           causticRadiance, globalPhotonCount, causticPhotonCount, false,
           causticCount, causticPhotonDir, causticPhotonPos, rng);
    
    if (causticCount > 0) {
      causticSample.photonPosition = causticPhotonPos;
      causticSample.photonDirection = causticPhotonDir;
      causticSample.photonFlux = causticRadiance;
      
      float causticLuminance = calcBt709Luminance(causticSample.photonFlux);
      if (causticLuminance > 0.0f) {
        // Store GI-style normalized weight (1 / area PDF).
        // pdf = 1 / (pi * r^2), so W = 1 / pdf = pi * r^2.
        float photonArea = pi * causticRadius * causticRadius;
        causticReservoir.targetFunc = causticLuminance;
        causticReservoir.wSum = photonArea;
        causticReservoir.M = 1;
        causticReservoir.pathLength = primaryIsSpecular ? 1 : 0;
      }
    }
    
    // Store surface data for temporal reuse using proper packed format
    ReSTIRFG_SurfaceData surfaceData;
    surfaceData.position = surfacePos;
    surfaceData.normal = surfaceNormal;
    surfaceData.roughness = roughness;
    surfaceData.albedo = float3(1); // TODO: read from GBuffer if available
    surfaceData.hitInfo = 1; // Mark as valid
    
    // Store directly as packed structure
    SurfaceDataOutput[linearIdx] = PackedReSTIRFG_SurfaceData::pack(surfaceData);
  } else {
    // Invalid surface - store empty surface data
    ReSTIRFG_SurfaceData emptySurface;
    emptySurface.position = float3(0);
    emptySurface.normal = float3(0, 1, 0);
    emptySurface.roughness = 1.0f;
    emptySurface.albedo = float3(0);
    emptySurface.hitInfo = 0; // Mark as invalid
    SurfaceDataOutput[linearIdx] = PackedReSTIRFG_SurfaceData::pack(emptySurface);
  }
  
  // Write outputs
  FGReservoirOutput[linearIdx] = PackedReSTIRFG_Reservoir::pack(fgReservoir);
  FGSampleOutput[linearIdx] = PackedReSTIRFG_Sample::pack(fgSample);
  CausticReservoirOutput[linearIdx] = PackedReSTIRFG_CausticReservoir::pack(causticReservoir);
  CausticSampleOutput[linearIdx] = PackedReSTIRFG_CausticSample::pack(causticSample);
}
