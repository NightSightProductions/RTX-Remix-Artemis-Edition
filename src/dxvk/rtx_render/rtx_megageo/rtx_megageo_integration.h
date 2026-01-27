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

#pragma once

#include "rtx_megageo_builder.h"
#include "../rtx_types.h"
#include "../rtx_materials.h"

namespace dxvk {

  /**
   * \brief RTX MegaGeometry Integration Helpers
   *
   * Helper functions to integrate RTX Remix materials, geometries, and
   * scene data with the RTX MG subdivision surface system.
   */
  class RtxMegaGeoIntegration {
  public:
    /**
     * \brief Check if geometry is suitable for subdivision
     *
     * Determines if a mesh should be treated as a subdivision surface based on:
     * - Topology (all quads)
     * - USD subdivision scheme attribute
     * - Material displacement/normal map presence
     * - Performance heuristics (poly count, screen coverage)
     *
     * \param [in] geometry RTX Remix geometry data
     * \param [in] material RTX Remix material data
     * \return true if mesh should use subdivision
     */
    static bool isSubdivisionCandidate(
      const RaytraceGeometry& geometry,
      const RtxMaterial& material);

    /**
     * \brief Extract subdivision surface from RTX Remix geometry
     *
     * Converts an RTX Remix RaytraceGeometry into a SubdivisionSurfaceDesc
     * for RTX MG processing. Extracts:
     * - Control points from vertex buffer
     * - Quad topology from index buffer
     * - Texture coordinates
     * - Material displacement parameters
     *
     * \param [in] geometry RTX Remix geometry
     * \param [in] material RTX Remix material
     * \param [out] desc Subdivision surface descriptor
     * \return true on successful extraction
     */
    static bool extractSubdivisionSurface(
      const RaytraceGeometry& geometry,
      const RtxMaterial& material,
      SubdivisionSurfaceDesc& desc);

    /**
     * \brief Configure displacement from RTX Remix material
     *
     * Extracts displacement/height map parameters from an RTX Remix material
     * and configures them for RTX MG. Handles:
     * - Height map textures
     * - Normal map displacement (converts to height)
     * - Displacement scale from material properties
     * - Bindless texture index assignment
     *
     * \param [in] material RTX Remix material
     * \param [out] desc Subdivision surface descriptor (updated with displacement params)
     * \return true if displacement is enabled for this material
     */
    static bool configureDisplacement(
      const RtxMaterial& material,
      SubdivisionSurfaceDesc& desc);

    /**
     * \brief Detect quad topology from triangle mesh
     *
     * Analyzes a triangle mesh to determine if it represents a subdivided quad mesh.
     * Looks for pairs of triangles sharing an edge that form quads.
     *
     * \param [in] indices Triangle mesh indices
     * \param [in] numIndices Number of indices (must be divisible by 3)
     * \param [out] quadIndices Output quad indices (4 per quad)
     * \param [out] numQuads Number of quads detected
     * \return true if mesh is all quads
     */
    static bool detectQuadTopology(
      const uint32_t* indices,
      uint32_t numIndices,
      std::vector<uint32_t>& quadIndices,
      uint32_t& numQuads);

    /**
     * \brief Calculate optimal isolation level
     *
     * Determines the best subdivision isolation level based on:
     * - Screen space coverage
     * - Distance from camera
     * - Geometric detail (curvature)
     * - Performance budget
     *
     * \param [in] geometry Geometry data
     * \param [in] worldToClip World-to-clip matrix
     * \param [in] viewportSize Viewport dimensions
     * \return Recommended isolation level (0-6)
     */
    static uint32_t calculateIsolationLevel(
      const RaytraceGeometry& geometry,
      const Matrix4& worldToClip,
      const Vector2& viewportSize);
  };

  /**
   * \brief RTX MegaGeometry Scene Manager
   *
   * Integrates RTX MG into the RTX Remix scene graph by:
   * - Detecting subdivision candidates during scene processing
   * - Building cluster BLAS for subdivision surfaces
   * - Injecting cluster BLAS into AccelManager
   * - Managing subdivision surface lifecycle
   */
  class RtxMegaGeoSceneManager {
  public:
    RtxMegaGeoSceneManager(
      const Rc<DxvkDevice>& device,
      const Rc<RtxContext>& context);

    /**
     * \brief Process draw call for potential subdivision
     *
     * Called during scene processing to check if a draw call should use
     * subdivision surfaces instead of standard tessellation.
     *
     * \param [in] drawCall Draw call state
     * \param [in] material Material data
     * \return true if draw call was handled by RTX MG
     */
    bool processDrawCall(
      const DrawCallState& drawCall,
      const RtxMaterial& material);

    /**
     * \brief Build all subdivision surfaces for the frame
     *
     * Executes the RTX MG pipeline for all active subdivision surfaces:
     * - Updates HiZ buffer
     * - Computes cluster tiling (tessellation + culling)
     * - Builds cluster BLAS
     * - Returns BLAS handles for TLAS construction
     *
     * \param [in] context RTX context
     * \param [in] depthBuffer Depth buffer for HiZ culling
     * \param [in] viewProj View-projection matrix
     * \return Number of BLAS built
     */
    uint32_t buildFrame(
      const Rc<RtxContext>& context,
      const Rc<DxvkImageView>& depthBuffer,
      const Matrix4& viewProj);

    /**
     * \brief Get subdivision BLAS for draw call
     *
     * Returns the cluster BLAS for a draw call that was processed
     * by RTX MG, for injection into the TLAS.
     *
     * \param [in] drawCallHash Hash of the draw call
     * \return BLAS handle or VK_NULL_HANDLE
     */
    VkAccelerationStructureKHR getBlas(XXH64_hash_t drawCallHash) const;

    /**
     * \brief Get BLAS device address
     *
     * \param [in] drawCallHash Hash of the draw call
     * \return Device address or 0
     */
    VkDeviceAddress getBlasAddress(XXH64_hash_t drawCallHash) const;

    /**
     * \brief Get statistics for the current frame
     */
    const RtxMegaGeoBuilder::TessellationStats& getStats() const {
      return m_builder->getStats();
    }

  private:
    Rc<DxvkDevice> m_device;
    Rc<RtxContext> m_context;
    Rc<RtxMegaGeoBuilder> m_builder;

    // Map draw call hash -> surface ID
    std::unordered_map<XXH64_hash_t, uint32_t> m_drawCallToSurface;

    // Pending surface builds for this frame
    std::vector<uint32_t> m_pendingSurfaces;
  };

} // namespace dxvk
