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

// ReSTIR-FG Photon Tracing Pass
// Traces photons from light sources and stores them for later collection

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_types.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/sampling.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/concept/ray/ray.slangh"
#include "rtx/concept/ray/ray_helper.slangh"
#include "rtx/concept/light/light.slangh"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/concept/surface/surface.slangh"
#include "rtx/concept/surface_material/surface_material.slangh"
#include "rtx/utility/debug_view_helpers.slangh"

// Outputs
layout(binding = RESTIR_FG_TRACE_BINDING_PHOTON_BUFFER_OUTPUT)
RWStructuredBuffer<PackedPhotonData> PhotonBuffer;

// AABB buffer as array of 6 floats (VkAabbPositionsKHR format: minX, minY, minZ, maxX, maxY, maxZ)
layout(binding = RESTIR_FG_TRACE_BINDING_PHOTON_AABB_GLOBAL_OUTPUT)
RWStructuredBuffer<float> PhotonAABBGlobal;

layout(binding = RESTIR_FG_TRACE_BINDING_PHOTON_AABB_CAUSTIC_OUTPUT)
RWStructuredBuffer<float> PhotonAABBCaustic;

layout(binding = RESTIR_FG_TRACE_BINDING_PHOTON_COUNTER)
RWStructuredBuffer<uint> PhotonCounter; // [0] = global count, [1] = caustic count

// Photon type constants
#define kPhotonType_Global 0
#define kPhotonType_Caustic 1

// Check if a surface is considered diffuse based on roughness
bool isDiffuseSurface(float roughness)
{
  return roughness >= cb.restirFGRoughnessThreshold;
}

// Store a photon in the buffer
void storePhoton(float3 position, float3 incomingDir, float3 flux, uint type)
{
  float radius = (type == kPhotonType_Global) ? cb.restirFGPhotonRadius : cb.restirFGCausticPhotonRadius;
  
  // Atomically increment the counter for this photon type
  uint photonIndex;
  InterlockedAdd(PhotonCounter[type], 1, photonIndex);
  
  // Check bounds
  if (photonIndex >= cb.restirFGPhotonsPerFrame) {
    return;
  }
  
  // Calculate buffer index (global photons first, then caustic)
  uint bufferIndex = (type == kPhotonType_Global) ? photonIndex : (cb.restirFGPhotonsPerFrame + photonIndex);
  
  // Create photon data
  PhotonData pd;
  pd.position = position;
  pd.radius = radius;
  pd.flux = flux;
  pd.dirPacked = packDirection(incomingDir);
  pd.photonType = type;
  
  // Store packed photon
  PhotonBuffer[bufferIndex] = PackedPhotonData::pack(pd);
  
  // Store AABB for acceleration structure building (VkAabbPositionsKHR: 6 floats)
  float3 minAABB = position - float3(radius);
  float3 maxAABB = position + float3(radius);
  
  uint aabbBase = photonIndex * 6;
  if (type == kPhotonType_Global) {
    PhotonAABBGlobal[aabbBase + 0] = minAABB.x;
    PhotonAABBGlobal[aabbBase + 1] = minAABB.y;
    PhotonAABBGlobal[aabbBase + 2] = minAABB.z;
    PhotonAABBGlobal[aabbBase + 3] = maxAABB.x;
    PhotonAABBGlobal[aabbBase + 4] = maxAABB.y;
    PhotonAABBGlobal[aabbBase + 5] = maxAABB.z;
  } else {
    PhotonAABBCaustic[aabbBase + 0] = minAABB.x;
    PhotonAABBCaustic[aabbBase + 1] = minAABB.y;
    PhotonAABBCaustic[aabbBase + 2] = minAABB.z;
    PhotonAABBCaustic[aabbBase + 3] = maxAABB.x;
    PhotonAABBCaustic[aabbBase + 4] = maxAABB.y;
    PhotonAABBCaustic[aabbBase + 5] = maxAABB.z;
  }
}

// Sample a light source and return emission parameters
bool sampleLightEmission(inout RNG rng, out float3 lightPos, out float3 lightRadiance, out f16vec3 lightNormal)
{
  uint lightCount = cb.lightCount;
  
  if (lightCount == 0) {
    return false;
  }
  
  // Randomly select a light
  uint lightIdx = uint(getNextSampleBlueNoise(rng) * float(lightCount)) % lightCount;
  
  // Load light data
  MemoryPolymorphicLight memoryLight = lights[lightIdx];
  DecodedPolymorphicLight decodedLight = decodePolymorphicLight(memoryLight);
  
  // Sample a point on the light
  float2 lightUV = float2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng));
  
  switch (decodedLight.polymorphicType) {
    case lightTypeSphere: {
      SphereLight sphereLight = sphereLightCreate(decodedLight);
      f16vec2 rndSample = f16vec2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng));
      float solidAnglePdf;
      f16vec3 sampleDir = calcCosineHemisphereDirectionSample(rndSample, solidAnglePdf);
      f16vec3 tangent, bitangent;
      f16vec3 upVec = f16vec3(0.0h, 1.0h, 0.0h);
      calcOrthonormalBasis(upVec, tangent, bitangent);
      sampleDir = tangent * sampleDir.x + upVec * sampleDir.y + bitangent * sampleDir.z;
      sampleDir = normalize(sampleDir);
      
      lightPos = sphereLight.position + float3(sampleDir) * float(sphereLight.radius);
      lightRadiance = sphereLight.radiance * float(lightCount);
      lightNormal = sampleDir;
      break;
    }
    case lightTypeRect: {
      RectLight rectLight = rectLightCreate(decodedLight);
      f16vec3 offset = rectLight.xAxis * (float16_t(lightUV.x - 0.5f) * rectLight.dimensions.x) 
                     + rectLight.yAxis * (float16_t(lightUV.y - 0.5f) * rectLight.dimensions.y);
      lightPos = rectLight.position + float3(offset);
      lightRadiance = rectLight.radiance * float(lightCount);
      lightNormal = rectLight.normal;
      break;
    }
    case lightTypeDisk: {
      DiskLight diskLight = diskLightCreate(decodedLight);
      float r = sqrt(lightUV.x);
      float theta = twoPi * lightUV.y;
      f16vec3 offset = diskLight.xAxis * float16_t(r * cos(theta) * float(diskLight.halfDimensions.x)) 
                     + diskLight.yAxis * float16_t(r * sin(theta) * float(diskLight.halfDimensions.y));
      lightPos = diskLight.position + float3(offset);
      lightRadiance = diskLight.radiance * float(lightCount);
      lightNormal = diskLight.normal;
      break;
    }
    case lightTypeDistant: {
      DistantLight distantLight = distantLightCreate(decodedLight);
      float3 sceneCenter = cameraGetWorldPosition(cb.camera); 
      lightPos = sceneCenter + float3(distantLight.direction) * 1000.0f;
      lightRadiance = distantLight.radiance * float(lightCount);
      lightNormal = -distantLight.direction;
      break;
    }
    default: {
      lightPos = cameraGetWorldPosition(cb.camera) + float3(0, 50, 0);
      lightRadiance = float3(1, 1, 1);
      lightNormal = f16vec3(0.0h, -1.0h, 0.0h);
      break;
    }
  }
  
  return true;
}

// Trace a photon path through the scene with multiple bounces
void tracePhoton(uint photonIndex, inout RNG rng)
{
  // Sample light emission
  float3 lightPos;
  float3 lightRadiance;
  f16vec3 lightNormal;
  
  if (!sampleLightEmission(rng, lightPos, lightRadiance, lightNormal)) {
    return;
  }
  
  // Generate emission direction (cosine-weighted from light normal)
  f16vec2 rndDir = f16vec2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng));
  float solidAnglePdf;
  f16vec3 localDir = calcCosineHemisphereDirectionSample(rndDir, solidAnglePdf);
  
  // Transform to world space using orthonormal basis from light normal
  f16vec3 tangent, bitangent;
  calcOrthonormalBasis(lightNormal, tangent, bitangent);
  f16vec3 emitDir = normalize(tangent * localDir.x + lightNormal * localDir.y + bitangent * localDir.z);
  
  // Initialize photon path
  float3 photonFlux = lightRadiance / float(cb.restirFGPhotonsPerFrame);
  float3 rayOrigin = lightPos;
  float3 rayDir = float3(emitDir);
  
  // Track if we've hit specular surfaces (for caustic photon classification)
  bool hasHitSpecular = false;
  uint maxBounces = cb.restirFGMaxPhotonBounces;
  
  // Trace photon through scene
  for (uint bounce = 0; bounce < maxBounces; bounce++) {
    // Russian roulette termination after first bounce
    if (bounce > 0) {
      float continueProbability = min(max(photonFlux.x, max(photonFlux.y, photonFlux.z)), 0.95f);
      if (getNextSampleBlueNoise(rng) > continueProbability) {
        break;
      }
      photonFlux /= continueProbability;
    }
    
    // Trace ray using rayQuery
    RayDesc rayDesc;
    rayDesc.Origin = rayOrigin;
    rayDesc.Direction = rayDir;
    rayDesc.TMin = 0.001f;
    rayDesc.TMax = 10000.0f;
    
    RayQuery<0> q;
    q.TraceRayInline(topLevelAS, 0, OBJECT_MASK_OPAQUE | OBJECT_MASK_TRANSLUCENT, rayDesc);
    
    while (q.Proceed()) {}
    
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
      break; // Ray escaped scene
    }
    
    // Get hit information using proper geometry fetching pattern
    RAY_HIT_INFO_CREATE_RAY_QUERY_COMMITTED(rayHitInfo, q)
    
    // Create ray for interaction
    Ray ray;
    ray.origin = rayOrigin;
    ray.direction = rayDir;
    ray.coneRadius = float16_t(0.0f);
    ray.spreadAngle = float16_t(0.001f);
    
    // Create ray interaction from hit info
    RayInteraction rayInteraction = rayInteractionCreate(ray, rayHitInfo);
    
    float hitT = rayHitInfo.hitDistance;
    float3 hitPos = rayOrigin + rayDir * hitT;
    
    // Fetch surface from surfaces buffer
    uint surfaceIdx = calculateSurfaceIndex(rayHitInfo);
    Surface surface = surfaces[surfaceIdx];
    
    // Create surface interaction to get proper geometric normal
    SurfaceInteraction surfaceInteraction = surfaceInteractionCreate<SurfaceIgnoreTangents>(
      surface, rayInteraction, ray, false, kFootprintFromRayDirection);
    
    // Use proper triangle normal
    float3 hitNormal = float3(surfaceInteraction.triangleNormal);
    
    // Fetch material to determine roughness
    const MemoryPolymorphicSurfaceMaterial memMaterial = surfaceMaterials[surfaceIdx];
    const uint8_t materialType = rayInteraction.materialType;
    
    float roughness = 0.5f; // Default for non-opaque materials
    bool forceSpecularTransport = false;
    float3 forcedTransportDir = float3(0.0f);
    float3 specularTransportThroughput = float3(1.0f);
    if (materialType == surfaceMaterialTypeOpaque) {
      const OpaqueSurfaceMaterial opaqueMaterial = opaqueSurfaceMaterialCreate(memMaterial, true);
      const OpaqueSurfaceMaterialInteraction materialInteraction = 
        opaqueSurfaceMaterialInteractionCreate(opaqueMaterial, surface, surfaceInteraction, rayInteraction);
      roughness = float(materialInteraction.isotropicRoughness);
    } else if (materialType == surfaceMaterialTypeTranslucent) {
      // Treat translucent interfaces as specular transport for caustic classification.
      const TranslucentSurfaceMaterial translucentMaterial = translucentSurfaceMaterialCreate(memMaterial, true);
      const TranslucentSurfaceMaterialInteraction translucentInteraction =
        translucentSurfaceMaterialInteractionCreate(translucentMaterial, surface, surfaceInteraction, rayInteraction);

      float ior = max(float(translucentInteraction.refractiveIndex), 1.001f);
      float3 transportNormal = hitNormal;
      float etaI = 1.0f;
      float etaT = ior;

      if (dot(rayDir, transportNormal) > 0.0f) {
        transportNormal = -transportNormal;
        float tmpEta = etaI;
        etaI = etaT;
        etaT = tmpEta;
      }

      float eta = etaI / max(etaT, 1e-4f);
      float3 refractedDir = refract(rayDir, transportNormal, eta);
      bool hasRefraction = dot(refractedDir, refractedDir) > 1e-8f;

      forceSpecularTransport = true;
      forcedTransportDir = hasRefraction ? refractedDir : reflect(rayDir, transportNormal);
      specularTransportThroughput = max(float3(translucentInteraction.transmittanceColor), float3(0.05f));
    }
    
    // Check if surface is diffuse
    bool isDiffuse = !forceSpecularTransport && (roughness >= cb.restirFGRoughnessThreshold);
    
    if (!isDiffuse) {
      // Specular surface - continue tracing
      hasHitSpecular = true;
      
      // Opaque specular uses reflection; translucent prefers transmission/refraction.
      rayDir = forceSpecularTransport ? normalize(forcedTransportDir) : reflect(rayDir, hitNormal);
      rayOrigin = hitPos;
      
      if (forceSpecularTransport) {
        photonFlux *= specularTransportThroughput;
      } else {
        // Apply fresnel attenuation (simplified)
        float fresnel = 0.04f + 0.96f * pow(1.0f - max(0.0f, dot(-rayDir, hitNormal)), 5.0f);
        photonFlux *= fresnel;
      }
      
      continue;
    }
    
    // Diffuse surface - store photon.
    // Optional global rejection increases relative caustic density in mixed scenes.
    uint photonType = hasHitSpecular ? kPhotonType_Caustic : kPhotonType_Global;
    bool storeThisPhoton = true;
    float3 storedFlux = photonFlux;
    if (photonType == kPhotonType_Global && cb.restirFGGlobalPhotonRejectionProbability > 0.0f) {
      float keepProbability = max(1.0f - cb.restirFGGlobalPhotonRejectionProbability, 1e-3f);
      storeThisPhoton = getNextSampleBlueNoise(rng) < keepProbability;
      if (storeThisPhoton) {
        // Preserve expected global-photon energy under stochastic rejection.
        storedFlux /= keepProbability;
      }
    }

    if (storeThisPhoton) {
      // Store photon transport direction as-traced (into the receiving surface).
      storePhoton(hitPos, rayDir, storedFlux, photonType);
    }
    
    // Continue tracing for diffuse bounces (for global illumination)
    if (bounce < maxBounces - 1 && !hasHitSpecular) {
      // Sample cosine-weighted hemisphere for diffuse bounce
      f16vec2 rndBounce = f16vec2(getNextSampleBlueNoise(rng), getNextSampleBlueNoise(rng));
      float bouncePdf;
      f16vec3 bounceLocalDir = calcCosineHemisphereDirectionSample(rndBounce, bouncePdf);
      
      // Build local frame from hit normal
      f16vec3 hitTangent, hitBitangent;
      calcOrthonormalBasis(f16vec3(hitNormal), hitTangent, hitBitangent);
      
      rayDir = float3(hitTangent * bounceLocalDir.x + f16vec3(hitNormal) * bounceLocalDir.y + hitBitangent * bounceLocalDir.z);
      rayOrigin = hitPos;
      
      // Attenuate flux by diffuse BRDF (assume 50% albedo)
      photonFlux *= 0.5f;
    } else {
      break; // Stop after storing caustic photon
    }
  }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint photonIndex = dispatchThreadId.x;
  
  if (photonIndex >= cb.restirFGPhotonsPerFrame) {
    return;
  }
  
  // Initialize RNG for random number generation
  RNG rng = createRNG(uvec2(photonIndex, 0), cb.frameIdx);
  
  // Trace the photon
  tracePhoton(photonIndex, rng);
}
