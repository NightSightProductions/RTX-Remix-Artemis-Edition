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
// Enable verbose MegaGeo logging for debugging (0=off for performance, 1=on for debugging)
#define RTXMG_VERBOSE_LOGGING 0
#if RTXMG_VERBOSE_LOGGING
#define RTXMG_LOG(msg) Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

#include "rtx_megageo_builder.h"
#include "../rtx_context.h"
#include "../rtx_camera.h"
#include "../../util/log/log.h"
#include "../../../util/util_error.h"  // For DxvkError exception handling
#include <cmath>
#include <chrono>

// Enable chrono timing for performance profiling (set to 1 to enable)
#define RTXMG_CHRONO_TIMING 1
#include "nvrhi_adapter/nvrhi_dxvk_texture.h"
#include "scene/rtxmg_scene.h"
#include "scene/instance.h"
#include "subdivision/subdivision_surface.h"
#include "subdivision/topology_cache.h"
#include "subdivision/shape.h"
#include "cluster_builder/fill_instance_descs_params.h"

// OpenSubdiv includes for topology building (using osd_lite like the sample)
#include "osd_lite/opensubdiv/far/topologyDescriptor.h"
#include "osd_lite/opensubdiv/far/topologyRefinerFactory.h"
#include "osd_lite/opensubdiv/sdc/types.h"
#include "osd_lite/opensubdiv/sdc/options.h"
#include "osd_lite/opensubdiv/tmr/topologyMap.h"
#include "osd_lite/opensubdiv/tmr/surfaceTableFactory.h"

namespace dxvk {

  // Use OpenSubdiv namespace (from osd_lite, matching the sample)
  // Only use the versioned namespace to avoid ambiguity
  using namespace OpenSubdiv::OPENSUBDIV_VERSION;

  RtxMegaGeoBuilder::RtxMegaGeoBuilder(
    const Rc<DxvkDevice>& device,
    const Rc<RtxContext>& rtxContext)
    : m_device(device)
    , m_rtxContext(rtxContext)
  {
  }

  RtxMegaGeoBuilder::~RtxMegaGeoBuilder() {
    // Shut down worker threads
    if (!m_workerThreads.empty()) {
      RTXMG_LOG(str::format("RTX MegaGeo: Shutting down ", m_workerThreads.size(), " worker threads"));
      m_workerShouldExit.store(true);
      m_workerCV.notify_all();
      for (auto& thread : m_workerThreads) {
        if (thread.joinable()) {
          thread.join();
        }
      }
      m_workerThreads.clear();
      RTXMG_LOG("RTX MegaGeo: All worker threads shut down");
    }

    // Cleanup BLAS handles
    for (auto& [id, surface] : m_surfaces) {
      if (surface.blas != VK_NULL_HANDLE) {
        // BLAS cleanup will be handled by DXVK's resource management
        surface.blas = VK_NULL_HANDLE;
      }
    }

    // NVRHI adapter cleanup
    if (m_nvrhiDevice) {
      delete m_nvrhiDevice;
      m_nvrhiDevice = nullptr;
    }
  }

  bool RtxMegaGeoBuilder::initialize() {
    if (m_initialized) {
      Logger::warn("RtxMegaGeoBuilder::initialize() called multiple times");
      return true;
    }

    Logger::info("Initializing RTX Mega Geometry Builder...");

    // Create NVRHI adapter
    m_nvrhiDevice = new NvrhiDxvkDevice(
      m_device,
      m_rtxContext,  // RtxContext inherits from DxvkContext
      m_rtxContext);

    if (!m_nvrhiDevice) {
      Logger::err("Failed to create NVRHI device adapter");
      return false;
    }

    // Create NVRHI command list
    m_commandList = m_nvrhiDevice->createCommandList();
    if (!m_commandList) {
      Logger::err("Failed to create NVRHI command list");
      return false;
    }

    // Create ClusterAccelBuilder
    RTXMG_LOG("RTX MegaGeo: Creating ClusterAccelBuilder...");
    RTXMG_LOG("RTX MegaGeo: Note: Cluster acceleration requires VK_NV_cluster_acceleration_structure extension");
    RTXMG_LOG("RTX MegaGeo: This extension is only available on NVIDIA RTX GPUs with latest drivers");

    try {
      m_clusterBuilder = std::make_unique<ClusterAccelBuilder>(m_nvrhiDevice, m_rtxContext.ptr());
      RTXMG_LOG("RTX MegaGeo: ClusterAccelBuilder created successfully");
    } catch (const std::exception& e) {
      Logger::err(str::format("ClusterAccelBuilder creation failed: ", e.what()));
      Logger::err("RTX MegaGeo: This likely means VK_NV_cluster_acceleration_structure extension is not available");
      return false;
    }

    // Create ClusterAccels storage
    m_clusterAccels = std::make_unique<ClusterAccels>();

    // Create RTXMGScene for managing subdivision surfaces and instances
    m_scene = std::make_unique<RTXMGScene>(m_nvrhiDevice);
    m_scene->Initialize();

    // Create TopologyCache for each worker thread (thread-safe parallel processing)
    TopologyCache::Options topoCacheOptions{
      .isoLevelSharp = 3,
      .isoLevelSmooth = 2,
      .useTerminalNodes = true
    };
    m_topologyCaches.reserve(m_numWorkerThreads);
    for (uint32_t i = 0; i < m_numWorkerThreads; ++i) {
      m_topologyCaches.push_back(std::make_unique<TopologyCache>(topoCacheOptions));
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Scene and ", m_numWorkerThreads, " topology caches initialized"));

    // Create HIZ buffer (for visibility culling)
    // Note: HIZ buffer will be created on-demand with appropriate size when needed
    // The static Create method requires size parameters that aren't known at init time

    // Create Z buffer (depth buffer interface)
    // Note: Z buffer will be created on-demand when needed
    // The static Create method requires size parameters that aren't known at init time

    // Create scratch buffer for cluster operations
    // Start with a reasonable default size (16 MB), will grow if needed
    const uint64_t initialScratchSize = 16 * 1024 * 1024;
    nvrhi::BufferDesc scratchDesc;
    scratchDesc.byteSize = initialScratchSize;
    scratchDesc.debugName = "RTX MG Cluster Scratch";
    scratchDesc.canHaveUAVs = true;
    scratchDesc.canHaveRawViews = true;
    scratchDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    scratchDesc.keepInitialState = true;
    m_scratchBuffer = m_nvrhiDevice->createBuffer(scratchDesc);

    if (!m_scratchBuffer) {
      Logger::err("RTX MegaGeo: Failed to create scratch buffer");
      return false;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Created scratch buffer (", initialScratchSize / 1024, " KB)"));

    // Start async worker threads for subdivision surface creation
    RTXMG_LOG(str::format("RTX MegaGeo: Starting ", m_numWorkerThreads, " worker threads"));
    m_workerThreads.reserve(m_numWorkerThreads);
    for (uint32_t i = 0; i < m_numWorkerThreads; ++i) {
      m_workerThreads.emplace_back(&RtxMegaGeoBuilder::workerThreadFunc, this, i);
    }

    m_initialized = true;
    RTXMG_LOG("RTX Mega Geometry Builder initialized successfully");
    return true;
  }

  void RtxMegaGeoBuilder::workerThreadFunc(uint32_t threadIndex) {
    RTXMG_LOG(str::format("RTX MegaGeo: Worker thread ", threadIndex, " started"));

    while (!m_workerShouldExit.load()) {
      PendingSurface pending;
      bool hasPending = false;

      {
        std::unique_lock<std::mutex> lock(m_pendingMutex);
        m_workerCV.wait(lock, [this] {
          return !m_pendingSurfaces.empty() || m_workerShouldExit.load();
        });

        if (m_workerShouldExit.load() && m_pendingSurfaces.empty()) {
          break;
        }

        if (!m_pendingSurfaces.empty()) {
          pending = std::move(m_pendingSurfaces.front());
          m_pendingSurfaces.pop();
          hasPending = true;
        }
      }

      if (hasPending) {
        uint32_t numFaces = pending.shape ? pending.shape->nvertsPerFace.size() : 0;
        RTXMG_LOG(str::format("RTX MegaGeo: Worker[", threadIndex, "] processing surface ", pending.surfaceId, " (", numFaces, " faces)"));

        try {
          // Just validate and return the Shape - SubdivisionSurface will be created on main thread
          // The Shape object already contains all the topology data needed
          RTXMG_LOG(str::format("RTX MegaGeo: Worker[", threadIndex, "] completed surface ", pending.surfaceId));

          // Move to completed queue
          {
            std::lock_guard<std::mutex> lock(m_completedMutex);
            m_completedSurfaces.push({
              pending.surfaceId,
              std::move(pending.shape),
              pending.topologyHash
            });
          }
        } catch (const std::exception& e) {
          Logger::err(str::format("RTX MegaGeo: Worker[", threadIndex, "] failed to process surface ", pending.surfaceId, ": ", e.what()));
        } catch (...) {
          Logger::err(str::format("RTX MegaGeo: Worker[", threadIndex, "] failed to process surface ", pending.surfaceId, ": unknown exception"));
        }
      }
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Worker thread ", threadIndex, " exiting"));
  }

  bool RtxMegaGeoBuilder::createSubdivisionSurface(
    const SubdivisionSurfaceDesc& desc,
    uint32_t& surfaceId)
  {
    if (!m_initialized) {
      Logger::err("RtxMegaGeoBuilder not initialized - call initialize() first");
      return false;
    }

    // Validate input
    if (desc.numFaces == 0 || desc.numVertices == 0) {
      Logger::err("Invalid subdivision surface: numFaces or numVertices is zero");
      return false;
    }

    if (!desc.faceVertexIndices || !desc.controlPoints) {
      Logger::err("Invalid subdivision surface: missing required data");
      return false;
    }

    // Allocate surface ID
    surfaceId = m_nextSurfaceId++;

    // Convert descriptor to Shape format
    auto shape = convertDescToShape(desc);
    if (!shape) {
      Logger::err(str::format("Failed to convert descriptor to Shape for surface ", surfaceId));
      return false;
    }

    // Queue for async creation on worker thread
    RTXMG_LOG(str::format("RTX MegaGeo: Queuing surface ", surfaceId, " for async creation (", desc.numFaces, " faces)"));

    {
      std::lock_guard<std::mutex> lock(m_pendingMutex);
      m_pendingSurfaces.push({
        surfaceId,
        std::move(shape),
        0 // topologyHash - caller can set this if needed
      });
    }
    m_workerCV.notify_one();

    // Create placeholder entry
    RTXMGSubdivisionSurfaceEntry surface;
    // Store only value-type data (raw pointers in desc become dangling after caller returns)
    surface.debugName = desc.debugName ? desc.debugName : "";
    surface.numVertices = desc.numVertices;
    surface.numFaces = desc.numFaces;
    surface.isolationLevel = desc.isolationLevel;
    surface.tessellationScale = desc.tessellationScale;
    surface.isDirty = true;
    surface.isReady = false; // Will be set to true when worker completes
    surface.m_hasDisplacementMaterial = desc.enableDisplacement;
    surface.displacementScale = desc.displacementScale;

    // Store placeholder in map
    m_surfaces[surfaceId] = std::move(surface);

    Logger::info(str::format("Queued subdivision surface ", surfaceId,
                             " for async creation (", desc.numFaces, " faces, ",
                             desc.numVertices, " vertices)"));

    return true;
  }

  bool RtxMegaGeoBuilder::updateSubdivisionSurface(
    uint32_t surfaceId,
    const SubdivisionSurfaceDesc& desc)
  {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      Logger::err(str::format("Surface ", surfaceId, " not found"));
      return false;
    }

    RTXMGSubdivisionSurfaceEntry& surface = it->second;

    surface.isDirty = true;
    surface.isReady = false;

    // Recreate Shape and SubdivisionSurface with updated data
    auto shape = convertDescToShape(desc);
    if (!shape) {
      Logger::err(str::format("Failed to convert descriptor to Shape for surface ", surfaceId));
      return false;
    }

    std::vector<std::unique_ptr<Shape>> keyFrames;
    surface.subdivSurface = std::make_unique<SubdivisionSurface>(
      *m_topologyCaches[0],
      std::move(shape),
      keyFrames,
      nullptr,
      m_commandList.Get());

    Logger::info(str::format("Updated subdivision surface ", surfaceId));
    return true;
  }

  void RtxMegaGeoBuilder::removeSubdivisionSurface(uint32_t surfaceId) {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return;
    }

    RTXMGSubdivisionSurfaceEntry& surface = it->second;

    // Cleanup BLAS
    if (surface.blas != VK_NULL_HANDLE) {
      // BLAS cleanup handled by DXVK
      surface.blas = VK_NULL_HANDLE;
    }

    // Remove from ClusterAccelBuilder
    // (ClusterAccelBuilder manages its own surface list)

    m_surfaces.erase(it);
    Logger::info(str::format("Removed subdivision surface ", surfaceId));
  }

  bool RtxMegaGeoBuilder::buildClusterBlas(
    const Rc<RtxContext>& context,
    const Rc<DxvkImageView>& depthBuffer,
    const RtCamera& rtCamera,
    const std::unordered_map<uint32_t, Matrix4>& instanceTransforms)
  {
    // Store instance transforms for use when setting localToWorld
    m_instanceTransforms = instanceTransforms;
    static uint32_t s_frameCounter = 0;
    s_frameCounter++;
    RTXMG_LOG(str::format("RTX MegaGeo: buildClusterBlas - FRAME ", s_frameCounter, " Entry"));

    // Reset scratch buffers at start of frame - DXVK ensures GPU is done with previous frame
    if (m_commandList) {
      RTXMG_LOG(str::format("RTX MegaGeo: FRAME ", s_frameCounter, " - calling clearState"));
      m_commandList->clearState();
      RTXMG_LOG(str::format("RTX MegaGeo: FRAME ", s_frameCounter, " - clearState done"));
    }

    if (!m_initialized) {
      Logger::err("RtxMegaGeoBuilder not initialized");
      return false;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: buildClusterBlas - Surface count: ", m_surfaces.size()));

    if (m_surfaces.empty()) {
      // No surfaces to tessellate
      return true;
    }

    // Update tessellation camera from RTX Remix camera
    // Use actual world space camera position for distance-based LOD calculations
    // Control points are stored in world space, so we need world space camera position
    dxvk::Vector3 camPos = rtCamera.getPosition(true);
    Vector3 actualCameraPos(camPos.x, camPos.y, camPos.z);
    m_tessellationCamera.SetEye(actualCameraPos);

    // Extract forward direction from view matrix for lookat
    Matrix4d worldToView = rtCamera.getWorldToView(true);
    float fwdX = -static_cast<float>(worldToView[0][2]);
    float fwdY = -static_cast<float>(worldToView[1][2]);
    float fwdZ = -static_cast<float>(worldToView[2][2]);
    Vector3 forward(fwdX, fwdY, fwdZ);
    m_tessellationCamera.SetLookat(actualCameraPos + forward);
    float upX = static_cast<float>(worldToView[0][1]);
    float upY = static_cast<float>(worldToView[1][1]);
    float upZ = static_cast<float>(worldToView[2][1]);
    Vector3 up(upX, upY, upZ);
    m_tessellationCamera.SetUp(up);
    // Extract FOV and aspect from projection matrix
    Matrix4d projMat = rtCamera.getViewToProjection();
    float proj11 = static_cast<float>(projMat[1][1]);
    float proj00 = static_cast<float>(projMat[0][0]);
    float fovY = 2.0f * std::atan(1.0f / proj11) * (180.0f / 3.14159265f);
    float aspectRatio = proj11 / proj00;
    m_tessellationCamera.SetFovY(fovY);
    m_tessellationCamera.SetAspectRatio(aspectRatio);
    // Get near/far planes directly from RtCamera - use actual game values
    float zNear = rtCamera.getNearPlane();
    float zFar = rtCamera.getFarPlane();
    // Only prevent degenerate values that would break the projection matrix
    if (zNear <= 0.0f) zNear = 0.001f;  // Tiny near plane to support close objects
    if (zFar <= zNear) zFar = 1000000.0f;  // Very large far plane - no artificial limit
    m_tessellationCamera.SetNear(zNear);
    m_tessellationCamera.SetFar(zFar);

    // CRITICAL: Initialize cluster templates BEFORE binding any image views (like HiZ)
    // The sync Downloads in template initialization close/reopen the command list
    // which destroys any bound resources. Must happen before updateHiZBuffer!
    {
      uint32_t maxGeometryCountPerMesh = m_scene ?
          static_cast<uint32_t>(m_scene->GetSceneGraph()->GetMaxGeometryCountPerMesh()) : 1;
      RTXMG_LOG(str::format("RTX MegaGeo: buildClusterBlas - Ensuring templates initialized, maxGeoPerMesh=", maxGeometryCountPerMesh));
      m_clusterBuilder->EnsureTemplatesInitialized(maxGeometryCountPerMesh, m_commandList.Get());
      RTXMG_LOG("RTX MegaGeo: buildClusterBlas - Templates initialized");
    }

    // Update HIZ buffer from depth buffer (if provided)
    // Skip first 2 frames - depth buffer isn't rendered to yet and has UNDEFINED layout
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - About to update HiZ buffer");
    if (depthBuffer != nullptr && s_frameCounter > 2) {
      updateHiZBuffer(depthBuffer);
    } else if (depthBuffer != nullptr && s_frameCounter <= 2) {
      RTXMG_LOG("RTX MegaGeo: Skipping HiZ update - waiting for depth buffer to be rendered");
    }
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - HiZ buffer update complete");

    // Configure tessellator
    TessellatorConfig config;
    config.enableVertexNormals = true;
    config.enableLogging = false;  // DISABLED - causes GPU readback stalls (~1 second per frame)
    config.zbuffer = m_zBuffer.get();
    config.camera = &m_tessellationCamera;

    // Set viewport size from depth buffer - CRITICAL for tessellation rate calculations
    // Without this, viewportSize is {0,0} and edge tessellation rates are all 0, causing clusters=0
    if (depthBuffer != nullptr) {
      VkExtent3D extent = depthBuffer->mipLevelExtent(0);
      config.viewportSize = { extent.width, extent.height };
      RTXMG_LOG(str::format("RTX MegaGeo: viewportSize set to ", extent.width, "x", extent.height));
    } else {
      // Fallback to a reasonable default if no depth buffer
      config.viewportSize = { 1920, 1080 };
      RTXMG_LOG("RTX MegaGeo: No depth buffer, using default viewportSize 1920x1080");
    }

    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - About to call BuildAccel");

    // Initialize TopologyCache GPU buffers (plansBuffer, subpatchTreesArraysBuffer, etc.)
    // This must be called before BuildAccel to ensure TopologyMap buffers are available
    // for FillInstanceClusters shader bindings
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - Initializing TopologyCache device data");
    m_topologyCaches[0]->InitDeviceData(m_commandList.Get());
    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - TopologyCache device data initialized");

    // PROPER FIX: Rebuild instance list each frame from active transforms
    // This ensures only surfaces with valid transforms this frame are rendered
    // Prevents "sticking to screen" bug from identity transforms on inactive surfaces
    RTXMG_LOG(str::format("RTX MegaGeo: Rebuilding instances - m_instanceTransforms.size()=", m_instanceTransforms.size(),
        " m_surfaceToMeshIndex.size()=", m_surfaceToMeshIndex.size()));

    if (m_scene) {
      // Clear instances from previous frame
      m_scene->ClearInstances();
      m_surfaceToInstanceIndex.clear();

      uint32_t instancesCreated = 0;
      uint32_t meshesNotReady = 0;

      // Rebuild instances only for surfaceIds that have transforms this frame
      for (const auto& [surfaceId, transform] : m_instanceTransforms) {
        auto meshIt = m_surfaceToMeshIndex.find(surfaceId);
        if (meshIt == m_surfaceToMeshIndex.end()) {
          // Mesh not ready yet (async creation in progress)
          meshesNotReady++;
          continue;
        }

        uint32_t meshIndex = meshIt->second;

        // Verify mesh exists
        if (meshIndex >= m_scene->GetSubdMeshes().size()) {
          Logger::warn(str::format("RTX MegaGeo: Invalid meshIndex ", meshIndex, " for surfaceId ", surfaceId));
          continue;
        }

        // Create instance for this surface with the proper transform
        Instance instance;
        instance.meshID = meshIndex;
        instance.localToWorld = transform;  // Use actual transform, NOT identity

        // Create MeshInfo for instance
        auto meshInfo = std::make_shared<MeshInfo>();
        SubdivisionSurface* mesh = m_scene->GetSubdMeshes()[meshIndex];
        const Shape* shape = mesh ? mesh->GetShape() : nullptr;
        meshInfo->name = shape ? shape->filepath.string() : "RTXMGSurface";
        meshInfo->geometryCount = 1;

        auto geomInfo = std::make_shared<GeometryInfo>();
        geomInfo->globalGeometryIndex = meshIndex;
        geomInfo->name = meshInfo->name;
        meshInfo->geometries.push_back(geomInfo);

        instance.meshInstance = std::make_shared<MeshInstance>(meshInfo);

        // Record mapping BEFORE adding (so instanceIndex is correct)
        uint32_t instanceIndex = static_cast<uint32_t>(m_scene->GetSubdMeshInstances().size());
        m_surfaceToInstanceIndex[surfaceId] = instanceIndex;

        m_scene->AddInstance(instance);
        instancesCreated++;

        // Log first few for debugging - show full matrix to verify layout
#if RTXMG_VERBOSE_LOGGING
        if (instancesCreated <= 3) {
          RTXMG_LOG(str::format("RTX MegaGeo: Created instance surfaceId=", surfaceId,
              " meshIdx=", meshIndex, " instanceIdx=", instanceIndex));
          // Log full matrix to understand layout (row-major: row3 = translation, column-major: col3 = translation)
          RTXMG_LOG(str::format("  row0=(", transform.data[0][0], ",", transform.data[0][1], ",", transform.data[0][2], ",", transform.data[0][3], ")"));
          RTXMG_LOG(str::format("  row1=(", transform.data[1][0], ",", transform.data[1][1], ",", transform.data[1][2], ",", transform.data[1][3], ")"));
          RTXMG_LOG(str::format("  row2=(", transform.data[2][0], ",", transform.data[2][1], ",", transform.data[2][2], ",", transform.data[2][3], ")"));
          RTXMG_LOG(str::format("  row3=(", transform.data[3][0], ",", transform.data[3][1], ",", transform.data[3][2], ",", transform.data[3][3], ")"));
          // Also show what col3 looks like (in case it's column-major)
          RTXMG_LOG(str::format("  col3=(", transform.data[0][3], ",", transform.data[1][3], ",", transform.data[2][3], ",", transform.data[3][3], ")"));
        }
#endif
      }

      RTXMG_LOG(str::format("RTX MegaGeo: Rebuilt ", instancesCreated, " instances (",
          meshesNotReady, " meshes not ready)"));
    } else {
      RTXMG_LOG("RTX MegaGeo: Scene is null, cannot rebuild instances");
    }

    // Chrono timing for BuildAccel
#if RTXMG_CHRONO_TIMING
    static float s_buildTimeMs = 0.0f;
    auto buildStart = std::chrono::high_resolution_clock::now();
#endif

    try {
      // Actually call ClusterAccelBuilder::BuildAccel
      // The command list wraps RTX Remix's DxvkContext, which is always "open"
      // Commands are recorded immediately and will be submitted by the rendering pipeline
      // Do NOT call open() or close() here - let RTX Remix manage command submission
      m_clusterBuilder->BuildAccel(*m_scene, config, *m_clusterAccels, m_clusterStats, m_frameIndex++, m_commandList.Get());
      RTXMG_LOG("RTX MegaGeo: buildClusterBlas - BuildAccel completed");
    } catch (const dxvk::DxvkError& e) {
      // DxvkError doesn't inherit from std::exception, so catch it separately
      Logger::err(str::format("RTX MegaGeo: Failed to build cluster BLAS: ", e.message()));
      Logger::err("RTX MegaGeo: This is likely a memory allocation failure - try reducing memory settings");
      return false;
    } catch (const std::exception& e) {
      Logger::err(str::format("RTX MegaGeo: Failed to build cluster BLAS: ", e.what()));
      Logger::err("RTX MegaGeo: This likely means VK_NV_cluster_acceleration_structure extension is not available");
      Logger::err("RTX MegaGeo: Cluster-based tessellation requires NVIDIA RTX GPU with latest drivers");
      return false;
    } catch (...) {
      Logger::err("RTX MegaGeo: Failed to build cluster BLAS: unknown error");
      return false;
    }

#if RTXMG_CHRONO_TIMING
    auto buildEnd = std::chrono::high_resolution_clock::now();
    auto buildDuration = std::chrono::duration_cast<std::chrono::microseconds>(buildEnd - buildStart);
    float frameTimeMs = buildDuration.count() * 0.001f;
    s_buildTimeMs = s_buildTimeMs * 0.9f + frameTimeMs * 0.1f; // Smoothed average
    Logger::info(str::format(">>> RTXMG CHRONO: BuildAccel frame=", frameTimeMs, "ms smoothed=", s_buildTimeMs, "ms"));
#endif

    // Mark all surfaces as ready
    // Note: BLAS addresses will be patched directly on GPU by AccelManager::patchClusterBlasAddresses()
    // The blasPtrsBuffer contains GPU-side addresses that will be copied to the instance buffer
    if (m_clusterAccels->blasBuffer) {
      RTXMG_LOG("RTX MegaGeo: Cluster BLAS built successfully");
      RTXMG_LOG(str::format("RTX MegaGeo: blasPtrsBuffer available: ", (m_clusterAccels->blasPtrsBuffer.Get() != nullptr)));

      // Log buffer sizes for debugging GPU crashes
      Logger::info(str::format("RTX MegaGeo: Buffer sizes - clusterShadingData: ",
          m_clusterAccels->clusterShadingDataBuffer.GetBytes(), " bytes (",
          m_clusterAccels->clusterShadingDataBuffer.GetNumElements(), " elements)"));
      Logger::info(str::format("RTX MegaGeo: Buffer sizes - clusterVertexPositions: ",
          m_clusterAccels->clusterVertexPositionsBuffer.GetBytes(), " bytes (",
          m_clusterAccels->clusterVertexPositionsBuffer.GetNumElements(), " elements)"));
      Logger::info(str::format("RTX MegaGeo: Buffer sizes - clusterVertexNormals: ",
          m_clusterAccels->clusterVertexNormalsBuffer.GetBytes(), " bytes (",
          m_clusterAccels->clusterVertexNormalsBuffer.GetNumElements(), " elements)"));
      Logger::info(str::format("RTX MegaGeo: numClusters=", m_clusterStats.allocated.m_numClusters,
          " numTriangles=", m_clusterStats.allocated.m_numTriangles));

      // Mark all surfaces as ready - BLAS addresses will be patched on GPU
      for (auto& [id, surface] : m_surfaces) {
        if (surface.isDirty && surface.subdivSurface) {
          surface.isDirty = false;
          surface.isReady = true;
          // Note: blasAddress is left as 0 - patching happens on GPU via patchClusterBlasAddresses()
          RTXMG_LOG(str::format("RTX MegaGeo: Surface ", id, " marked ready"));
        }
      }
    }

    // Collect statistics from ClusterStatistics
    m_stats.numClusters = m_clusterStats.allocated.m_numClusters;
    m_stats.numDesiredClusters = m_clusterStats.desired.m_numClusters;
    m_stats.numTriangles = m_clusterStats.allocated.m_numTriangles;
    m_stats.numVertices = 0; // Not directly available in ClusterStatistics
    m_stats.clasMemoryBytes = m_clusterStats.allocated.m_clasSize;
    m_stats.cullRatio = (m_stats.numDesiredClusters > 0) ?
      1.0f - (float)m_stats.numClusters / (float)m_stats.numDesiredClusters : 0.0f;

    RTXMG_LOG(str::format("RTX MegaGeo: Built cluster BLAS: ", m_stats.numClusters, " clusters, ",
                            m_stats.numTriangles, " triangles"));

    // Always log RTXMG stats prominently so user can verify it's working
    static uint32_t frameLogCounter = 0;
    if ((frameLogCounter++ % 60) == 0) { // Log every 60 frames (~1 second)
      if (m_stats.numClusters > 0) {
#if RTXMG_CHRONO_TIMING
        Logger::info(str::format(">>> RTXMG: ", m_stats.numClusters, " clusters, ",
                                  m_stats.numTriangles, " tris, ", m_surfaces.size(), " surfaces, ",
                                  s_buildTimeMs, "ms <<<"));
#else
        Logger::info(str::format(">>> RTXMG ACTIVE: ", m_stats.numClusters, " clusters, ",
                                  m_stats.numTriangles, " triangles from ", m_surfaces.size(), " surfaces <<<"));
#endif
      } else {
        Logger::warn(str::format(">>> RTXMG: 0 clusters (desired: ", m_stats.numDesiredClusters, ") from ", m_surfaces.size(), " surfaces - CHECK TESSELLATION <<<"));
      }
    }

    RTXMG_LOG("RTX MegaGeo: buildClusterBlas - returning true");
    return true;
  }

  VkAccelerationStructureKHR RtxMegaGeoBuilder::getSurfaceBlas(uint32_t surfaceId) const {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return VK_NULL_HANDLE;
    }
    return it->second.blas;
  }

  VkDeviceAddress RtxMegaGeoBuilder::getSurfaceBlasAddress(uint32_t surfaceId) const {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return 0;
    }
    return it->second.blasAddress;
  }

  bool RtxMegaGeoBuilder::isSurfaceReady(uint32_t surfaceId) const {
    auto it = m_surfaces.find(surfaceId);
    if (it == m_surfaces.end()) {
      return false;
    }
    return it->second.isReady;
  }

  nvrhi::IBuffer* RtxMegaGeoBuilder::getBlasPointersBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->blasPtrsBuffer.Get();
  }

  uint32_t RtxMegaGeoBuilder::getInstanceIndexForSurface(uint32_t surfaceId) const {
    auto it = m_surfaceToInstanceIndex.find(surfaceId);
    if (it == m_surfaceToInstanceIndex.end()) {
      return UINT32_MAX;
    }
    return it->second;
  }

  nvrhi::BufferHandle RtxMegaGeoBuilder::getClusterShadingDataBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->clusterShadingDataBuffer.Get();
  }

  uint32_t RtxMegaGeoBuilder::getClusterCount() const {
    return m_stats.numClusters;
  }

  nvrhi::BufferHandle RtxMegaGeoBuilder::getClusterVertexPositionsBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->clusterVertexPositionsBuffer.Get();
  }

  nvrhi::BufferHandle RtxMegaGeoBuilder::getClusterVertexNormalsBuffer() const {
    if (!m_clusterAccels) {
      return nullptr;
    }
    return m_clusterAccels->clusterVertexNormalsBuffer.Get();
  }

  void RtxMegaGeoBuilder::processCompletedSurfaces() {
    std::queue<CompletedSurface> completed;

    {
      std::lock_guard<std::mutex> lock(m_completedMutex);
      completed.swap(m_completedSurfaces);
    }

    while (!completed.empty()) {
      CompletedSurface& comp = completed.front();

      auto it = m_surfaces.find(comp.surfaceId);
      if (it != m_surfaces.end()) {
        RTXMGSubdivisionSurfaceEntry& surface = it->second;

        RTXMG_LOG(str::format("RTX MegaGeo: Creating SubdivisionSurface for surface ", comp.surfaceId, " on main thread"));

        // Create SubdivisionSurface on main thread with proper GPU resources
        m_commandList->open();

        try {
          std::vector<std::unique_ptr<Shape>> keyFrames;
          surface.subdivSurface = std::make_unique<SubdivisionSurface>(
            *m_topologyCaches[0], // Use first topology cache on main thread
            std::move(comp.shape),
            keyFrames,
            nullptr, // descriptorTableManager - may need to provide this
            m_commandList.Get());

          m_commandList->close();
          m_nvrhiDevice->executeCommandList(m_commandList.Get());

          // Add to RTXMGScene - get the actual index for meshID
          uint32_t meshIndex = static_cast<uint32_t>(m_scene->GetSubdMeshes().size());
          m_scene->AddSubdMesh(surface.subdivSurface.get());

          // Record the PERSISTENT mapping from surfaceId to mesh index
          // Instances will be rebuilt each frame in buildClusterBlas based on active transforms
          m_surfaceToMeshIndex[comp.surfaceId] = meshIndex;
          RTXMG_LOG(str::format("RTX MegaGeo: Mapping surfaceId ", comp.surfaceId, " to mesh index ", meshIndex));

          surface.isReady = true;

          const Shape* shape = surface.subdivSurface ? surface.subdivSurface->GetShape() : nullptr;
          uint32_t numFaces = shape ? shape->nvertsPerFace.size() : 0;
          RTXMG_LOG(str::format("RTX MegaGeo: Integrated completed surface ", comp.surfaceId,
                                   " (", numFaces, " faces)"));
        } catch (const std::exception& e) {
          m_commandList->close();
          Logger::err(str::format("RTX MegaGeo: Failed to create SubdivisionSurface for surface ", comp.surfaceId, ": ", e.what()));
        }
      }

      completed.pop();
    }
  }

  std::unique_ptr<Shape> RtxMegaGeoBuilder::convertDescToShape(const SubdivisionSurfaceDesc& desc)
  {
    RTXMG_LOG("RTX MegaGeo: Converting SubdivisionSurfaceDesc to Shape");

    // Validate input pointers
    if (!desc.controlPoints || !desc.faceVertexIndices || !desc.faceVertexCounts) {
      Logger::err("RTX MegaGeo: Invalid desc - null pointers");
      return nullptr;
    }

    // Log first few values to verify data is valid
    if (desc.numVertices > 0) {
      const Vector3& firstVert = desc.controlPoints[0];
      RTXMG_LOG(str::format("RTX MegaGeo: First vertex: (", firstVert.x, ", ", firstVert.y, ", ", firstVert.z, ")"));
    }
    if (desc.numFaces > 0 && desc.faceVertexCounts[0] > 0) {
      uint32_t firstIndex = desc.faceVertexIndices[0];
      RTXMG_LOG(str::format("RTX MegaGeo: First face index: ", firstIndex));
    }

    auto shape = std::make_unique<Shape>();

    // Set scheme to Catmull-Clark
    shape->scheme = Scheme::kCatmark;

    // Set filepath as name (Shape doesn't have a separate name field)
    if (desc.debugName) {
      shape->filepath = desc.debugName;
    }

    // Copy vertex positions
    shape->verts.resize(desc.numVertices);
    for (uint32_t i = 0; i < desc.numVertices; ++i) {
      shape->verts[i] = desc.controlPoints[i];
    }
    RTXMG_LOG(str::format("RTX MegaGeo: Copied ", desc.numVertices, " vertices"));

    // Copy UVs if present
    if (desc.texcoords) {
      shape->uvs.resize(desc.numVertices);
      for (uint32_t i = 0; i < desc.numVertices; ++i) {
        shape->uvs[i] = desc.texcoords[i];
      }
    }

    // Build face vertex counts and indices
    shape->nvertsPerFace.resize(desc.numFaces);
    uint32_t totalIndices = 0;
    for (uint32_t i = 0; i < desc.numFaces; ++i) {
      shape->nvertsPerFace[i] = static_cast<int>(desc.faceVertexCounts[i]);
      totalIndices += desc.faceVertexCounts[i];
    }

    shape->faceverts.resize(totalIndices);
    for (uint32_t i = 0; i < totalIndices; ++i) {
      uint32_t vertexIndex = desc.faceVertexIndices[i];

      // Validate vertex index is within bounds - fail immediately if invalid
      if (vertexIndex >= desc.numVertices) {
        Logger::err(str::format("RTX MegaGeo: Invalid vertex index ", vertexIndex,
                                " at position ", i, " (numVertices=", desc.numVertices, ")"));
        Logger::err("RTX MegaGeo: This indicates corrupted mesh data or incorrect index buffer format");
        return nullptr;
      }

      shape->faceverts[i] = static_cast<int>(vertexIndex);
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Copied ", totalIndices, " face indices"));

    // Setup face UV indices (faceuvs) if UVs are present
    // Since UVs are per-vertex, face UV indices are the same as face vertex indices
    if (desc.texcoords) {
      shape->faceuvs = shape->faceverts; // Copy the same indices
    }

    // Setup material bindings (required by OpenSubdiv)
    shape->mtlbind.resize(desc.numFaces, 0); // All faces use material 0
    shape->mtls.push_back(std::make_unique<Shape::material>()); // Create default material

    // Create single subshape for all faces
    Shape::Subshape subshape;
    subshape.startFaceIndex = 0;
    subshape.mtlBind = 0; // Single material
    shape->subshapes.push_back(subshape);

    // Map all faces to subshape 0
    shape->faceToSubshapeIndex.resize(desc.numFaces, 0);

    // Compute bounding box
    if (desc.numVertices > 0 && desc.controlPoints != nullptr) {
      Vector3 minPt = desc.controlPoints[0];
      Vector3 maxPt = desc.controlPoints[0];
      for (uint32_t i = 1; i < desc.numVertices; ++i) {
        const Vector3& p = desc.controlPoints[i];
        minPt.x = std::min(minPt.x, p.x);
        minPt.y = std::min(minPt.y, p.y);
        minPt.z = std::min(minPt.z, p.z);
        maxPt.x = std::max(maxPt.x, p.x);
        maxPt.y = std::max(maxPt.y, p.y);
        maxPt.z = std::max(maxPt.z, p.z);
      }
      shape->aabb.m_mins[0] = minPt.x;
      shape->aabb.m_mins[1] = minPt.y;
      shape->aabb.m_mins[2] = minPt.z;
      shape->aabb.m_maxs[0] = maxPt.x;
      shape->aabb.m_maxs[1] = maxPt.y;
      shape->aabb.m_maxs[2] = maxPt.z;
      Logger::info(str::format("convertToShape: Computed aabb min=(", minPt.x, ",", minPt.y, ",", minPt.z,
          ") max=(", maxPt.x, ",", maxPt.y, ",", maxPt.z, ")"));
    } else {
      Logger::warn(str::format("convertToShape: Cannot compute aabb - numVertices=", desc.numVertices,
          " controlPoints=", (void*)desc.controlPoints));
    }

    Logger::info(str::format("Converted to Shape: ", desc.numFaces, " faces, ", desc.numVertices, " vertices"));
    return shape;
  }


  void RtxMegaGeoBuilder::updateHiZBuffer(const Rc<DxvkImageView>& depthBuffer) {
    // Integrate RTX Remix depth buffer with RTX MG HiZ system
    // This follows the SDK pattern from zbuffer.cpp:154-161

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Entry");

    if (!m_commandList) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - No command list, returning");
      return;
    }

    if (depthBuffer == nullptr) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - No depth buffer, returning");
      return;
    }

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Getting image info");
    const Rc<DxvkImage>& image = depthBuffer->image();
    const DxvkImageCreateInfo& imageInfo = image->info();

    // Check if depth buffer has valid content (not UNDEFINED layout)
    if (imageInfo.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Depth buffer layout is UNDEFINED, skipping");
      return;
    }

    // Create ZBuffer on first use (when we know the depth buffer size)
    if (!m_zBuffer) {
      RTXMG_LOG(str::format("RTX MegaGeo: Creating ZBuffer with size ",
        imageInfo.extent.width, "x", imageInfo.extent.height));

      uint2 bufferSize = { imageInfo.extent.width, imageInfo.extent.height };

      // Create ZBuffer following reference sample pattern from zbuffer.cpp:85-119
      RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Calling ZBuffer::Create");
      m_zBuffer = ZBuffer::Create(bufferSize, m_nvrhiDevice, m_commandList.Get());

      if (!m_zBuffer) {
        Logger::err("RTX MegaGeo: Failed to create ZBuffer");
        return;
      }

      RTXMG_LOG("RTX MegaGeo: ZBuffer and HiZ hierarchy created successfully");
    }

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Creating scoped marker");
    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeo::updateHiZBuffer");

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Creating texture wrapper");
    // Copy depth buffer to ZBuffer current texture
    // Note: ZBuffer doesn't have Read() method in our adapter, so we directly copy
    nvrhi::TextureDesc depthDesc;
    depthDesc.width = imageInfo.extent.width;
    depthDesc.height = imageInfo.extent.height;
    depthDesc.depth = imageInfo.extent.depth;
    depthDesc.arraySize = imageInfo.numLayers;
    depthDesc.mipLevels = imageInfo.mipLevels;
    depthDesc.format = static_cast<nvrhi::Format>(imageInfo.format);
    depthDesc.debugName = "RTX Remix Depth Buffer";

    nvrhi::TextureHandle depthTexture = new dxvk::NvrhiDxvkTexture(depthDesc, image);

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Copying texture");
    // Copy to ZBuffer's current texture if available
    nvrhi::TextureHandle zbufferTex = m_zBuffer->GetCurrent();
    if (zbufferTex) {
      m_commandList->copyTexture(zbufferTex.Get(), depthTexture.Get());
    }

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Reducing HiZ hierarchy");
    // Reduce ZBuffer hierarchy to build HiZ mip chain
    // This performs min/max reduction across mip levels for efficient culling
    m_zBuffer->ReduceHierarchy(m_commandList.Get());

    RTXMG_LOG("RTX MegaGeo: updateHiZBuffer - Complete");
    // The HiZ buffer is now ready for use in compute_cluster_tiling shader
    // It will be bound via the TessellatorConfig::zbuffer pointer
  }


  void RtxMegaGeoBuilder::showImguiSettings() {
#ifdef IMGUI_ENABLED
    static const char* kColorModeNames[] = {
      "Base Color",
      "Surface Normal",
      "Tex Coord",
      "Material",
      "Geometry Index",
      "Surface Index",
      "Cluster ID",
      "MicroTri ID",
      "Cluster UV",
      "MicroTri Area",
      "Topology Quality"
    };
    static_assert(std::size(kColorModeNames) == static_cast<size_t>(ColorMode::COLOR_MODE_COUNT),
                  "ColorMode names must match enum count");

    ImGui::Checkbox("Enable Tessellation", &m_enableTessellation);
    if (!m_enableTessellation) {
      ImGui::TextDisabled("(Subdivision surfaces disabled)");
      return;
    }

    ImGui::Separator();

    // Micro triangle visualization
    if (ImGui::Checkbox("Show Micro Triangles", &m_showMicroTriangles)) {
      // When micro triangles are shown, override color mode
      if (m_showMicroTriangles) {
        m_colorMode = ColorMode::COLOR_BY_MICROTRI_ID;
      }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Toggle micro triangle visualization mode with a unique color per triangle id.");
    }

    // Wireframe mode
    ImGui::Checkbox("Wireframe", &m_wireframeMode);
    if (m_wireframeMode) {
      ImGui::SliderFloat("Wireframe Thickness", &m_wireframeThickness, 0.0f, 5.0f, "%.1f");
    }

    // Color mode selection (disabled when showing micro triangles)
    if (!m_showMicroTriangles) {
      int colorModeInt = static_cast<int>(m_colorMode);
      if (ImGui::Combo("Color Mode", &colorModeInt, kColorModeNames, static_cast<int>(ColorMode::COLOR_MODE_COUNT))) {
        m_colorMode = static_cast<ColorMode>(colorModeInt);
      }
    }

    ImGui::Separator();
    ImGui::Text("Tessellation Statistics:");

    // Display statistics
    ImGui::Text("Surfaces: %u", static_cast<uint32_t>(m_surfaces.size()));
    ImGui::Text("Clusters: %u / %u", m_stats.numClusters, m_stats.numDesiredClusters);
    ImGui::Text("Triangles: %u", m_stats.numTriangles);
    ImGui::Text("Vertices: %u", m_stats.numVertices);

    // Memory usage
    const float clasMemoryMB = m_stats.clasMemoryBytes / (1024.0f * 1024.0f);
    ImGui::Text("CLAS Memory: %.2f MB", clasMemoryMB);

    // Culling ratio
    if (m_stats.cullRatio > 0.0f) {
      ImGui::Text("Cull Ratio: %.1f%%", m_stats.cullRatio * 100.0f);
    }

    // Per-surface statistics
    if (ImGui::TreeNode("Per-Surface Stats")) {
      for (const auto& [id, surface] : m_surfaces) {
        std::string label = str::format("Surface ", id, ": ", surface.debugName.empty() ? "unnamed" : surface.debugName.c_str());
        if (ImGui::TreeNode(label.c_str())) {
          ImGui::Text("Vertices: %u", surface.numVertices);
          ImGui::Text("Faces: %u", surface.numFaces);
          ImGui::Text("Isolation Level: %u", surface.isolationLevel);
          ImGui::Text("Tessellation Scale: %.2f", surface.tessellationScale);
          ImGui::Text("Status: %s", surface.isReady ? "Ready" : (surface.isDirty ? "Dirty" : "Building"));
          if (surface.m_hasDisplacementMaterial) {
            ImGui::Text("Displacement: Enabled (scale=%.2f)", surface.displacementScale);
          }
          ImGui::TreePop();
        }
      }
      ImGui::TreePop();
    }
#endif
  }

  void RtxMegaGeoBuilder::patchClusterBlasAddresses(
    VkBuffer instanceBuffer,
    VkDeviceSize instanceBufferOffset,
    const std::vector<InstancePatchMapping>& mappings)
  {
    if (mappings.empty()) {
      RTXMG_LOG("RTX MegaGeo: patchClusterBlasAddresses - no mappings to patch");
      return;
    }

    if (!m_clusterAccels || m_clusterAccels->blasPtrsBuffer.GetBytes() == 0) {
      Logger::warn("RTX MegaGeo: patchClusterBlasAddresses - no blasPtrsBuffer available");
      return;
    }

    if (!m_nvrhiDevice || !m_commandList) {
      Logger::err("RTX MegaGeo: patchClusterBlasAddresses - NVRHI not initialized");
      return;
    }

    if (instanceBuffer == VK_NULL_HANDLE) {
      Logger::err("RTX MegaGeo: patchClusterBlasAddresses - null instance buffer");
      return;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: GPU-side patching ", mappings.size(), " cluster BLAS addresses"));

    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeoBuilder::patchClusterBlasAddresses");

    // Create or resize mappings buffer
    const size_t mappingsSize = mappings.size() * sizeof(ClusterInstanceMapping);
    if (!m_patchMappingsBuffer || m_patchMappingsBuffer->getDesc().byteSize < mappingsSize) {
      nvrhi::BufferDesc mappingsDesc;
      mappingsDesc.byteSize = mappingsSize;
      mappingsDesc.structStride = sizeof(ClusterInstanceMapping);
      mappingsDesc.debugName = "ClusterInstanceMappings";
      mappingsDesc.initialState = nvrhi::ResourceStates::ShaderResource;
      mappingsDesc.keepInitialState = true;
      m_patchMappingsBuffer = m_nvrhiDevice->createBuffer(mappingsDesc);
    }

    // Upload mappings to GPU
    std::vector<ClusterInstanceMapping> gpuMappings(mappings.size());
    for (size_t i = 0; i < mappings.size(); ++i) {
      gpuMappings[i].remixInstanceIndex = mappings[i].remixInstanceIndex;
      gpuMappings[i].rtxmgInstanceIndex = mappings[i].rtxmgInstanceIndex;
    }
    m_commandList->writeBuffer(m_patchMappingsBuffer, gpuMappings.data(), mappingsSize);

    // Create wrapper for the instance buffer
    // We need to create a temporary wrapper that references the external VkBuffer
    nvrhi::BufferDesc instanceBufDesc;
    instanceBufDesc.byteSize = mappings.size() * sizeof(VkAccelerationStructureInstanceKHR) * 4; // Conservative
    instanceBufDesc.structStride = sizeof(VkAccelerationStructureInstanceKHR);
    instanceBufDesc.debugName = "InstanceBufferWrapper";
    instanceBufDesc.canHaveUAVs = true;
    instanceBufDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;

    // Get the DxvkBuffer from the VkBuffer handle
    // Since we have the raw VkBuffer, we need to use it directly via the device address
    // Create a binding set that uses the blasPtrsBuffer and dispatches with device addresses

    // Set up binding set
    // Bindings:
    //   t0 = mappings buffer (SRV)
    //   t1 = blasPtrsBuffer (SRV)
    //   u0 = instance buffer (UAV) - need to create wrapper

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_patchMappingsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterAccels->blasPtrsBuffer.Get()));

    // For the UAV, we need the actual instance buffer wrapped
    // Since we only have VkBuffer, we'll need to get the DxvkBuffer from AccelManager
    // For now, fall back to CPU download approach but mark this as TODO for full GPU optimization

    // FALLBACK: Download and store for CPU-side patching
    // This is temporary until we can properly wrap the instance buffer
    std::vector<nvrhi::GpuVirtualAddress> blasAddresses = m_clusterAccels->blasPtrsBuffer.Download(m_commandList.Get());

    if (blasAddresses.empty()) {
      Logger::err("RTX MegaGeo: Failed to download BLAS addresses from GPU");
      return;
    }

    // Store downloaded addresses for use by AccelManager
    m_downloadedBlasAddresses.resize(blasAddresses.size());
    for (size_t i = 0; i < blasAddresses.size(); ++i) {
      m_downloadedBlasAddresses[i] = static_cast<VkDeviceAddress>(blasAddresses[i]);
    }

    RTXMG_LOG(str::format("RTX MegaGeo: Downloaded ", blasAddresses.size(), " BLAS addresses"));

    // Log first few addresses for debugging
    for (size_t i = 0; i < std::min<size_t>(3, blasAddresses.size()); ++i) {
      RTXMG_LOG(str::format("RTX MegaGeo: BLAS[", i, "] = 0x", std::hex, m_downloadedBlasAddresses[i]));
    }
  }

  void RtxMegaGeoBuilder::patchClusterBlasAddressesGPU(
    nvrhi::IBuffer* instanceBuffer,
    uint32_t instanceBufferOffset,
    const std::vector<InstancePatchMapping>& mappings)
  {
    if (mappings.empty()) {
      return;
    }

    if (!m_clusterAccels || m_clusterAccels->blasPtrsBuffer.GetBytes() == 0) {
      Logger::warn("RTX MegaGeo: patchClusterBlasAddressesGPU - no blasPtrsBuffer available");
      return;
    }

    if (!m_nvrhiDevice || !m_commandList) {
      Logger::err("RTX MegaGeo: patchClusterBlasAddressesGPU - NVRHI not initialized");
      return;
    }

    RTXMG_LOG(str::format("RTX MegaGeo: GPU-side patching ", mappings.size(), " cluster BLAS addresses"));

    nvrhi::utils::ScopedMarker marker(m_commandList.Get(), "RtxMegaGeoBuilder::patchClusterBlasAddressesGPU");

    // Create params buffer on first use
    if (!m_patchParamsBuffer) {
      nvrhi::BufferDesc paramsDesc;
      paramsDesc.byteSize = 256; // Align to 256 for constant buffer requirements
      paramsDesc.debugName = "PatchClusterBlasAddressParams";
      paramsDesc.isConstantBuffer = true;
      paramsDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
      paramsDesc.keepInitialState = true;
      m_patchParamsBuffer = m_nvrhiDevice->createBuffer(paramsDesc);
    }

    // Create or resize mappings buffer
    const size_t mappingsSize = mappings.size() * sizeof(ClusterInstanceMapping);
    if (!m_patchMappingsBuffer || m_patchMappingsBuffer->getDesc().byteSize < mappingsSize) {
      nvrhi::BufferDesc mappingsDesc;
      mappingsDesc.byteSize = mappingsSize;
      mappingsDesc.structStride = sizeof(ClusterInstanceMapping);
      mappingsDesc.debugName = "ClusterInstanceMappings";
      mappingsDesc.initialState = nvrhi::ResourceStates::ShaderResource;
      mappingsDesc.keepInitialState = true;
      m_patchMappingsBuffer = m_nvrhiDevice->createBuffer(mappingsDesc);
    }

    // Upload mappings to GPU
    std::vector<ClusterInstanceMapping> gpuMappings(mappings.size());
    for (size_t i = 0; i < mappings.size(); ++i) {
      gpuMappings[i].remixInstanceIndex = mappings[i].remixInstanceIndex + instanceBufferOffset;
      gpuMappings[i].rtxmgInstanceIndex = mappings[i].rtxmgInstanceIndex;
    }
    m_commandList->writeBuffer(m_patchMappingsBuffer, gpuMappings.data(), mappingsSize);

    // Write params to constant buffer
    PatchClusterBlasAddressParams params = {};
    params.numMappings = static_cast<uint32_t>(mappings.size());
    params.instanceBufferStride = sizeof(VkAccelerationStructureInstanceKHR);
    m_commandList->writeBuffer(m_patchParamsBuffer, &params, sizeof(params));

    // Set up binding set
    // Note: u0 uses StructuredBuffer_UAV to match the Slang shader's RWStructuredBuffer<VkInstanceRaw>
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_patchMappingsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterAccels->blasPtrsBuffer.Get()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, instanceBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_patchParamsBuffer));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(m_nvrhiDevice, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_patchBlasAddressesBL, bindingSet)) {
      Logger::err("RTX MegaGeo: Failed to create binding set for patch_cluster_blas_addresses");
      return;
    }

    // Create compute pipeline on first use
    if (!m_patchBlasAddressesPSO) {
      nvrhi::ShaderHandle shader = m_clusterBuilder->getShaderFactory().CreateShader(
        "cluster_builder/patch_cluster_blas_addresses.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

      if (!shader) {
        Logger::err("RTX MegaGeo: Failed to create patch_cluster_blas_addresses shader");
        return;
      }

      auto pipelineDesc = nvrhi::ComputePipelineDesc()
          .setComputeShader(shader)
          .addBindingLayout(m_patchBlasAddressesBL);

      m_patchBlasAddressesPSO = m_nvrhiDevice->createComputePipeline(pipelineDesc);
      if (!m_patchBlasAddressesPSO) {
        Logger::err("RTX MegaGeo: Failed to create patch_cluster_blas_addresses pipeline");
        return;
      }
    }

    // Set compute state and dispatch
    auto state = nvrhi::ComputeState()
        .setPipeline(m_patchBlasAddressesPSO)
        .addBindingSet(bindingSet);

    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - Setting compute state"));
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - blasPtrsBuffer bytes=", m_clusterAccels->blasPtrsBuffer.GetBytes(),
        " numElements=", m_clusterAccels->blasPtrsBuffer.GetNumElements()));
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - instanceBuffer byteSize=", instanceBuffer->getDesc().byteSize));

    m_commandList->setComputeState(state);

    // Dispatch - one thread per mapping
    uint32_t numGroups = (params.numMappings + kFillInstanceDescsThreads - 1) / kFillInstanceDescsThreads;
    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - dispatching numGroups=", numGroups, " for ", params.numMappings, " mappings"));
    m_commandList->dispatch(numGroups, 1, 1);

    RTXMG_LOG(str::format("RTX MegaGeo: patchClusterBlasAddressesGPU - dispatch complete for ", mappings.size(), " instances"));
  }

  VkDeviceAddress RtxMegaGeoBuilder::getDownloadedBlasAddress(uint32_t rtxmgInstanceIndex) const {
    if (rtxmgInstanceIndex >= m_downloadedBlasAddresses.size()) {
      return 0;
    }
    return m_downloadedBlasAddresses[rtxmgInstanceIndex];
  }

} // namespace dxvk
