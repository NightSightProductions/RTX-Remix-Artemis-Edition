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

// ReSTIR-FG Final Shading Pass
// Evaluates Final Gather and Caustic reservoirs to produce final indirect lighting

#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/restir_fg/restir_fg_binding_indices.h"
#include "rtx/pass/restir_fg/restir_fg_types.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/concept/camera/camera.slangh"

// G-buffer inputs
layout(rgba32f, binding = RESTIR_FG_FINAL_SHADING_BINDING_WORLD_POSITION_INPUT)
Texture2D<float4> WorldPosition;

layout(r32ui, binding = RESTIR_FG_FINAL_SHADING_BINDING_WORLD_NORMAL_INPUT)
Texture2D<uint> WorldNormal;

layout(rgba16f, binding = RESTIR_FG_FINAL_SHADING_BINDING_ALBEDO_INPUT)
Texture2D<float4> Albedo;

layout(r16f, binding = RESTIR_FG_FINAL_SHADING_BINDING_PERCEPTUAL_ROUGHNESS_INPUT)
Texture2D<float> PerceptualRoughness;

// Reservoir inputs
layout(binding = RESTIR_FG_FINAL_SHADING_BINDING_FG_RESERVOIR)
StructuredBuffer<PackedReSTIRFG_Reservoir> FGReservoir;

layout(binding = RESTIR_FG_FINAL_SHADING_BINDING_FG_SAMPLE)
StructuredBuffer<PackedReSTIRFG_Sample> FGSample;

layout(binding = RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_RESERVOIR)
StructuredBuffer<PackedReSTIRFG_CausticReservoir> CausticReservoir;

layout(binding = RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_SAMPLE)
StructuredBuffer<PackedReSTIRFG_CausticSample> CausticSample;

// Output - writes to the standard indirect radiance buffer for compositor integration
layout(rgba16f, binding = RESTIR_FG_FINAL_SHADING_BINDING_OUTPUT)
RWTexture2D<float4> PrimaryIndirectDiffuseRadianceHitDistance;

// Specular/Caustic output
layout(rgba16f, binding = RESTIR_FG_FINAL_SHADING_BINDING_CAUSTIC_OUTPUT)
RWTexture2D<float4> PrimaryIndirectSpecularRadianceHitDistance;

// Decode shading normal from packed format
float3 decodeNormal(uint packed)
{
  return signedOctahedralToSphereDirection(snorm2x16ToFloat2x32(packed));
}

// Evaluate diffuse BRDF (Lambertian)
float3 evaluateDiffuseBRDF(float3 albedo, float3 normal, float3 lightDir)
{
  float NdotL = max(0.0f, dot(normal, lightDir));
  return albedo * NdotL / pi;
}

// Get ReSTIR weight for final contribution
// Note: After resampling, wSum is already normalized to W (= wSum / (targetFunc * M))
// so we just use it directly - no further division needed!
float getReSTIRWeight(ReSTIRFG_Reservoir reservoir)
{
  if (reservoir.M == 0)
    return 0.0f;
  
  // wSum is already the final normalized weight W
  float w = reservoir.wSum;
  if (!isfinite(w) || w <= 0.0f)
    return 0.0f;
  return w;
}

float clampWeightBySampleLuminance(float weight, float3 sampleRadiance, bool isEmissiveSample)
{
  if (!isfinite(weight) || weight <= 0.0f)
    return 0.0f;

  float sampleLum = calcBt709Luminance(sampleRadiance);
  if (!isfinite(sampleLum) || sampleLum <= 0.0f)
    return 0.0f;

  float maxLuminance = cb.restirFGMaxLuminance;
  float clampLuminance = isEmissiveSample ? (maxLuminance * 16.0f) : maxLuminance;
  float maxWeight = clampLuminance / max(sampleLum, 1e-6f);

  return min(weight, max(maxWeight, 0.0f));
}

// Empty pixel hit distance constant
static const float kEmptyPixelHitDistance = 0.0f;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
  uint2 pixel = dispatchThreadId.xy;
  uint width, height;
  WorldPosition.GetDimensions(width, height);
  const uint2 resolution = uint2(width, height);
  
  if (any(pixel >= resolution)) {
    return;
  }
  
  uint linearIdx = pixel.y * resolution.x + pixel.x;
  
  // Load surface data from G-buffer
  float4 worldPosData = WorldPosition[pixel];
  float3 surfacePos = worldPosData.xyz;
  
  // Check for miss (invalid position)
  bool isValidSurface = any(surfacePos != float3(0.0f));
  
  if (!isValidSurface) {
    PrimaryIndirectDiffuseRadianceHitDistance[pixel] = float4(0.0f, 0.0f, 0.0f, kEmptyPixelHitDistance);
    PrimaryIndirectSpecularRadianceHitDistance[pixel] = float4(0.0f, 0.0f, 0.0f, kEmptyPixelHitDistance);
    return;
  }
  
  uint packedNormal = WorldNormal[pixel];
  float3 surfaceNormal = decodeNormal(packedNormal);
  
  float4 albedoData = Albedo[pixel];
  float3 albedo = albedoData.rgb;
  
  float roughness = PerceptualRoughness[pixel];
  
  // Camera view direction for specular
  float3 cameraPos = cameraGetWorldPosition(cb.camera);
  float3 viewDir = normalize(cameraPos - surfacePos);
  
  // Load reservoirs and samples
  ReSTIRFG_Reservoir fgReservoir = FGReservoir[linearIdx].unpack();
  ReSTIRFG_Sample fgSample = FGSample[linearIdx].unpack();
  ReSTIRFG_CausticReservoir causticReservoir = CausticReservoir[linearIdx].unpack();
  ReSTIRFG_CausticSample causticSample = CausticSample[linearIdx].unpack();
  
  // Initialize output radiance
  float3 indirectDiffuseRadiance = float3(0.0f);
  float3 specularRadiance = float3(0.0f);
  float hitDistance = kEmptyPixelHitDistance;
  
  // Evaluate Final Gather reservoir contribution
  // Per ReSTIR-FG full version: compute direction to FG sample and evaluate BSDF
  if (fgReservoir.M > 0 && fgReservoir.wSum > 0.0f) {
    // Get ReSTIR weight (already normalized in resample pass)
    float weight = getReSTIRWeight(fgReservoir);
    weight = clampWeightBySampleLuminance(weight, fgSample.radiance, fgSample.surfaceHit == 3u);
    
    if (weight > 0.0f && any(fgSample.radiance > float3(0.0f))) {
      // Per ReSTIR-FG paper (full version):
      // 1. Compute direction from shading surface TO the FG sample position
      float3 toFGSample = fgSample.hitPosition - surfacePos;
      float distToSample = length(toFGSample);
      float3 dir = toFGSample / max(distToSample, 1e-6f);
      
      // 2. The stored radiance is Li = collected photon flux at FG hit
      float3 Li = fgSample.radiance;
      
      // 3. Evaluate BSDF at the shading surface with direction to FG sample
      // For diffuse: f_r = albedo / pi, contribution = f_r * Li * NdotL
      float NdotL = max(0.0f, dot(surfaceNormal, dir));
      float3 diffuseBRDF = albedo / pi;
      float3 giDiff = diffuseBRDF * Li * NdotL;
      
      // 4. For glossy surfaces, add specular component
      float3 giSpec = float3(0.0f);
      if (roughness < 0.8f) {
        float3 H = normalize(viewDir + dir);
        float NdotH = max(0.0f, dot(surfaceNormal, H));
        float NdotV = max(0.0f, dot(surfaceNormal, viewDir));
        float VdotH = max(0.0f, dot(viewDir, H));
        
        // GGX NDF
        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        float denom = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
        float D = alpha2 / (pi * denom * denom + 1e-6f);
        
        // Simplified Fresnel (Schlick)
        float F = 0.04f + 0.96f * pow(1.0f - VdotH, 5.0f);
        
        // Visibility term (Smith GGX)
        float k = alpha / 2.0f;
        float G1_V = NdotV / (NdotV * (1.0f - k) + k + 1e-6f);
        float G1_L = NdotL / (NdotL * (1.0f - k) + k + 1e-6f);
        float G = G1_V * G1_L;
        
        giSpec = Li * (D * F * G) / (4.0f * NdotV + 1e-6f);
      }
      
      // 5. Final contribution with ReSTIR weight
      indirectDiffuseRadiance = giDiff * weight;
      specularRadiance += giSpec * weight;
      
      // Compute hit distance for denoiser
      hitDistance = max(distToSample, 0.001f);
    }
  }
  
  // Evaluate Caustic reservoir contribution
  // Caustics contribute to both specular surfaces and diffuse surfaces (viewed specular paths)
  if (causticReservoir.M > 0 && causticReservoir.wSum > 0.0f) {
    // wSum is already normalized to W in caustic resample pass
    float causticWeight = causticReservoir.wSum;
    if (!isfinite(causticWeight) || causticWeight <= 0.0f)
      causticWeight = 0.0f;
    causticWeight = clampWeightBySampleLuminance(causticWeight, causticSample.photonFlux, false);
    
    if (causticWeight > 0.0f && any(causticSample.photonFlux > float3(0.0f))) {
      // Photon direction is the incoming light direction (stored as the direction the photon was traveling)
      float3 photonDir = normalize(causticSample.photonDirection);
      
      // For caustics on diffuse surfaces, use Lambertian BRDF
      // NdotL = -dot(surfaceNormal, photonDir) since photonDir points INTO the surface
      float NdotL = max(0.0f, dot(surfaceNormal, -photonDir));
      
      // Caustic radiance: photon flux modulated by surface BRDF
      // The photon flux already includes the specular path contribution,
      // we just apply the diffuse BRDF at the final surface
      float3 causticContribution = causticSample.photonFlux * albedo * NdotL / pi * causticWeight;
      
      // Add caustics to both diffuse (for diffuse surfaces) and specular (for glossy surfaces)
      // Most caustics appear on diffuse surfaces from specular transport
      indirectDiffuseRadiance += causticContribution;
      
      // For specular surfaces, also add a glossy caustic term
      if (roughness < 0.5f) {
        float3 halfVec = normalize(viewDir - photonDir);
        float NdotH = max(0.0f, dot(surfaceNormal, halfVec));
        
        float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
        float denom = NdotH * NdotH * (alpha2 - 1.0f) + 1.0f;
        float D = alpha2 / (pi * denom * denom + 1e-6f);
        
        specularRadiance += causticSample.photonFlux * D * causticWeight * 0.1f;
      }
      
      // Update hit distance if needed
      if (hitDistance <= kEmptyPixelHitDistance) {
        hitDistance = max(length(causticSample.photonPosition - surfacePos), 0.001f);
      }
    }
  }
  
  // Clamp to prevent fireflies
  float maxLuminance = cb.restirFGMaxLuminance;
  const bool isEmissiveSample = (fgSample.surfaceHit == 3u);
  float clampLuminance = isEmissiveSample ? (maxLuminance * 16.0f) : maxLuminance;
  float diffuseLum = calcBt709Luminance(indirectDiffuseRadiance);
  if (diffuseLum > clampLuminance) {
    indirectDiffuseRadiance *= clampLuminance / diffuseLum;
  }
  
  float specularLum = calcBt709Luminance(specularRadiance);
  if (specularLum > clampLuminance) {
    specularRadiance *= clampLuminance / specularLum;
  }
  
  // Write outputs with hit distance for denoiser
  PrimaryIndirectDiffuseRadianceHitDistance[pixel] = float4(indirectDiffuseRadiance, hitDistance);
  PrimaryIndirectSpecularRadianceHitDistance[pixel] = float4(specularRadiance, hitDistance);
}
