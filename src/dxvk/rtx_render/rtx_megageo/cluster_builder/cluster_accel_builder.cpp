// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Enable verbose MegaGeo logging for debugging (0=off for performance, 1=on for debugging)
#define RTXMG_VERBOSE_LOGGING 0
#if RTXMG_VERBOSE_LOGGING
#define RTXMG_LOG(msg) Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif

// Enable chrono timing for performance profiling (set to 1 to enable)
#define RTXMG_CHRONO_TIMING 1

// Helper to align buffer sizes to 4 bytes (required for Vulkan vkCmdUpdateBuffer)
inline size_t alignBufferSize(size_t size) {
    return (size + 3) & ~3;
}

// STL includes
#include <algorithm>
#include <limits>
#include <chrono>

// RTX Remix includes
#include "../../../util/log/log.h"
#include "../../../util/util_string.h"
#include "../nvrhi_adapter/nvrhi_types.h"
#include "../nvrhi_adapter/nvrhi_dxvk_device.h"
#include "../nvrhi_adapter/nvrhi_dxvk_command_list.h"
#include "../nvrhi_adapter/nvrhi_dxvk_buffer.h"
#include "../nvrhi_adapter/nvrhi_dxvk_texture.h"
#include "../../rtx_shader_manager.h"
#include "../../rtx_context.h"

// RTX MG shader includes
#include <rtx_shaders/copy_cluster_offset.h>
#include <rtx_shaders/fill_instantiate_template_args.h>
#include <rtx_shaders/fill_blas_from_clas_args.h>
#include <rtx_shaders/fill_instance_descs.h>

// RTX MG shader binding indices
#include <rtx/pass/rtx_megageo/cluster_builder/copy_cluster_offset_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_clusters_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_instantiate_template_args_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_blas_from_clas_args_binding_indices.h>
#include <rtx/pass/rtx_megageo/cluster_builder/fill_instance_descs_binding_indices.h>

#include <map>
#include <fstream>

// RTX MG includes - updated paths
#include "cluster_accels.h"
#include "cluster_accel_builder.h"
#include "fill_clusters_params.h"
#include "copy_cluster_offset_params.h"
#include "fill_blas_from_clas_args_params.h"
#include "fill_instantiate_template_args_params.h"
#include "compute_cluster_tiling_params.h"
#include "tessellation_counters.h"
#include "tessellator_config.h"
#include "../scene/rtxmg_scene.h"
#include "../scene/instance.h"
#include "../scene/camera.h"

using namespace dxvk;
#include "tessellator_constants.h"

#include "../utils/buffer.h"
#include "../hiz/zbuffer.h"
#include "../hiz/hiz_buffer_constants.h"
#include "../profiler/profiler_stub.h"  // Lightweight profiler stub for RTX Remix

#include "../subdivision/subdivision_surface.h"
#include "../subdivision/topology_map.h"

using namespace donut;
using namespace nvrhi::rt;

constexpr uint32_t kNumTemplates = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments;
constexpr uint32_t kClusterMaxTriangles = kMaxClusterEdgeSegments * kMaxClusterEdgeSegments * 2;
constexpr uint32_t kClusterMaxVertices = (kMaxClusterEdgeSegments + 1) * (kMaxClusterEdgeSegments + 1);
constexpr uint32_t kFrameCount = 4;

ClusterAccelBuilder::ClusterAccelBuilder(
    nvrhi::DeviceHandle device,
    dxvk::RtxContext* rtxContext)
    : m_device(device)
    , m_rtxContext(rtxContext)
    , m_shaderFactory(rtxContext)
    , m_commonPasses(std::make_shared<donut::engine::CommonRenderPasses>(device))
{
    m_tessellationCountersBuffer.Create(kFrameCount, "tesselation counters", m_device.Get());
    m_debugBuffer.Create(512, "ClusterAccelDebug", m_device.Get());

    //////////////////////////////////////////////////
    // Parameter buffers for shaders
    //////////////////////////////////////////////////
    m_fillInstantiateTemplateArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillInstantiateTemplateArgsParams), "FillInstantiateTemplateArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_computeClusterTilingParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(ComputeClusterTilingParams), "ComputeClusterTilingParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillClustersParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillClustersParams), "FillClustersParams", engine::c_MaxRenderPassConstantBufferVersions));

    m_fillBlasFromClasArgsParamsBuffer = m_device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
        sizeof(FillBlasFromClasArgsParams), "FillBlasFromClasArgsParams", engine::c_MaxRenderPassConstantBufferVersions));

    //////////////////////////////////////////////////
    // Create common bindless binding layout and descriptor table
    //////////////////////////////////////////////////
    nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
    bindlessLayoutDesc.visibility = nvrhi::ShaderType::All;
    bindlessLayoutDesc.firstSlot = 0;
    bindlessLayoutDesc.maxCapacity = 1024;
    bindlessLayoutDesc.layoutType = nvrhi::BindlessLayoutDesc::LayoutType::MutableSrvUavCbv;
    m_bindlessBL = m_device->createBindlessLayout(bindlessLayoutDesc);

    // Create descriptor table (empty binding set) for the bindless layout.
    // This satisfies pipeline binding requirements even when no displacement maps are used.
    // When displacement is enabled, this would need to be populated with texture descriptors.
    m_descriptorTable = m_device->createDescriptorTable(m_bindlessBL);

    //////////////////////////////////////////////////
    // Create dummy HiZ textures for when HiZ culling is disabled
    // The shader expects HIZ_MAX_LODS textures at binding set 1, so we need
    // to bind valid textures even when HiZ is disabled to avoid validation errors
    //////////////////////////////////////////////////
    nvrhi::TextureDesc dummyHiZDesc;
    dummyHiZDesc.width = 1;
    dummyHiZDesc.height = 1;
    dummyHiZDesc.format = nvrhi::Format::R32_FLOAT;
    dummyHiZDesc.isUAV = true;  // Must be UAV so image is in GENERAL layout (matches bindHiZDescriptorSet expectations)
    dummyHiZDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    dummyHiZDesc.keepInitialState = true;

    for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
    {
        std::string debugName = "DummyHiZ_Level_" + std::to_string(i);
        dummyHiZDesc.debugName = debugName.c_str();
        m_dummyHiZTextures[i] = m_device->createTexture(dummyHiZDesc);
    }
}

// Must match shader defines in compute_cluster_tiling.hlsl
inline char const* toString(TessellatorConfig::AdaptiveTessellationMode mode)
{
    switch (mode)
    {
    case TessellatorConfig::AdaptiveTessellationMode::UNIFORM: return "TESS_MODE_UNIFORM";
    case TessellatorConfig::AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH: return "TESS_MODE_WORLD_SPACE_EDGE_LENGTH";
    case TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION: return "TESS_MODE_SPHERICAL_PROJECTION";
    default: return "UNKNOWN";
    }
}

inline char const* toString(TessellatorConfig::VisibilityMode mode)
{
    switch (mode)
    {
    case TessellatorConfig::VisibilityMode::VIS_SURFACE: return "VIS_MODE_SURFACE";
    case TessellatorConfig::VisibilityMode::VIS_LIMIT_EDGES: return "VIS_MODE_LIMIT_EDGES";
    default: return "UNKNOWN";
    }
}

constexpr auto kSurfaceTypeDefines = std::to_array<const char*>(
{
    "SURFACE_TYPE_PUREBSPLINE",
    "SURFACE_TYPE_REGULARBSPLINE",
    "SURFACE_TYPE_LIMIT",
    "SURFACE_TYPE_ALL"
});
static_assert(kSurfaceTypeDefines.size() == size_t(ShaderPermutationSurfaceType::Count));

inline char const* toString(ShaderPermutationSurfaceType surfaceType)
{
    return kSurfaceTypeDefines[uint32_t(surfaceType)];
}


void ClusterAccelBuilder::FillInstantiateTemplateArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* templateAddresses, uint32_t numTemplates, nvrhi::ICommandList* commandList)
{
    FillInstantiateTemplateArgsParams params = {};
    params.numTemplates = numTemplates;
    params.pad = uint3();

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::FillInstantiateTemplateArgs");
    commandList->writeBuffer(m_fillInstantiateTemplateArgsParamsBuffer, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, templateAddresses))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outArgs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillInstantiateTemplateArgsParamsBuffer));

    // Create layout once, then reuse for all binding sets (avoids CreateBindingSetAndLayout overhead)
    if (!m_fillInstantiateTemplateBL)
    {
        auto layoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
        m_fillInstantiateTemplateBL = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_fillInstantiateTemplateBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create binding set for fill_instantiate_template_args.hlsl");
    }

    if (!m_fillInstantiateTemplatePSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_instantiate_template_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillInstantiateTemplateBL);

        m_fillInstantiateTemplatePSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillInstantiateTemplatePSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numTemplates, kFillInstantiateTemplateArgsThreads), 1, 1);
}

void ClusterAccelBuilder::FillBlasFromClasArgs(nvrhi::IBuffer* outArgs, nvrhi::IBuffer* clusterOffsets,
    nvrhi::GpuVirtualAddress clasPtrsBaseAddress, uint32_t numInstances, nvrhi::ICommandList* commandList)
{
    FillBlasFromClasArgsParams params = {};
    params.clasAddressesBaseAddress = clasPtrsBaseAddress;
    params.numInstances = numInstances;

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::FillBlasFromClasArgs");
    commandList->writeBuffer(m_fillBlasFromClasArgsParamsBuffer, &params, sizeof(params));

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, clusterOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, outArgs))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillBlasFromClasArgsParamsBuffer));

    // Create layout once, then reuse for all binding sets
    if (!m_fillBlasFromClasArgsBL)
    {
        auto layoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
        m_fillBlasFromClasArgsBL = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_fillBlasFromClasArgsBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create binding set for fill_blas_from_clas_args.hlsl");
    }

    if (!m_fillBlasFromClasArgsPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_blas_from_clas_args.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_fillBlasFromClasArgsBL);

        m_fillBlasFromClasArgsPSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_fillBlasFromClasArgsPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(div_ceil(numInstances, kFillBlasFromClasArgsThreads), 1, 1);
}

static TemplateGrids GenerateTemplateGrids()
{
    TemplateGrids result;

    // Offsets per template
    result.descs.resize(kNumTemplates);
    result.indices.reserve(kNumTemplates * kClusterMaxTriangles * 3);
    result.vertices.reserve(kNumTemplates * kClusterMaxVertices * 3);

    // Generate cluster topologies for 11x11 grid
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        assert(i % kMaxClusterEdgeSegments < std::numeric_limits<TemplateGrids::IndexType>::max());
        assert(i / kMaxClusterEdgeSegments < std::numeric_limits<TemplateGrids::IndexType>::max());

        TemplateGridDesc& gridDesc = result.descs[i];
        gridDesc =
        {
            .xEdges = i % kMaxClusterEdgeSegments + 1,
            .yEdges = i / kMaxClusterEdgeSegments + 1,
            .indexOffset = static_cast<uint32_t>(result.indices.size() * sizeof(result.indices[0])),
            .vertexOffset = static_cast<uint32_t>(result.vertices.size() * sizeof(result.vertices[0]))
        };

        // x, y = lower - left vertex of quad
        // s: 0 is the first triangle, (left vertical edge), 1 is the second triangle (right vertical edge)
        auto TriIndices = [&gridDesc](uint32_t x, uint32_t y, uint32_t s)->std::array<uint32_t, 3>
        {
            uint32_t vs = gridDesc.getXVerts();  // vertex stride  (same as xVerts)
            uint32_t vid = y * vs + x;          // lower-left vertex id
            bool diag03 = ((x & 1) == (y & 1)); // is this triangle a 0-3 diagonal (true) or a 1-2 diagonal (false)

            assert(vid + vs + 1 < std::numeric_limits<TemplateGrids::IndexType>::max());

            // Example output for (xEdges = 3, yEdges = 1, x = {0..3}, y = 0)
            //      4_____5_____6_____7
            //      |    /|\    |    /|
            //      | a / | \ d | e / |
            //      |  /  |  \  |  /  |
            //      | / b | c \ | / f |
            //      |/____|____\|/____|
            //      0     1     2     3
            //
            //    (x,y,s)
            // a  (0,0,0) diag03 = (0, 5, 4) 
            // b  (0,0,1) diag03 = (0, 1, 5)
            // c  (1,0,0)        = (1, 2, 5)
            // d  (1,0,1)        = (2, 6, 5)
            // e  (2,0,0) diag03 = (2, 7, 6)
            // f  (2,0,1) diag03 = (2, 3, 7)

            if (diag03)
            {
                if (s == 0) return { vid, vid + 1 + vs, vid + vs };
                else        return { vid, vid + 1     , vid + 1 + vs };
            }
            else
            {
                if (s == 0) return { vid    , vid + 1     , vid + vs };
                else        return { vid + 1u, vid + 1 + vs, vid + vs };
            }
        };

        float xScale = 1.0f / gridDesc.xEdges;
        float yScale = 1.0f / gridDesc.yEdges;

        uint32_t xVerts = gridDesc.getXVerts();
        uint32_t yVerts = gridDesc.getYVerts();

        for (uint32_t y = 0; y < yVerts; y++)
        {
            for (uint32_t x = 0; x < xVerts; x++)
            {
                // Add triangles to index buffer
                if (x < gridDesc.xEdges && y < gridDesc.yEdges)
                {
                    for (uint32_t s = 0; s < 2; s++)
                    {
                        std::array<uint32_t, 3> triIndices = TriIndices(x, y, s);
                        std::transform(triIndices.begin(), triIndices.end(), std::back_inserter(result.indices), [](uint32_t e)
                            {
                                assert(e < std::numeric_limits<TemplateGrids::IndexType>::max());
                                return static_cast<TemplateGrids::IndexType>(e);
                            });
                    }
                }

                // Add verts
                result.vertices.push_back(x * xScale);
                result.vertices.push_back(y * yScale);
                result.vertices.push_back(0.0f);
            }
        }

        result.maxTriangles = std::max(result.maxTriangles, gridDesc.getNumTriangles());
        result.totalTriangles += gridDesc.getNumTriangles();

        result.maxVertices = std::max(result.maxVertices, gridDesc.getNumVerts());
        result.totalVertices += gridDesc.getNumVerts();
    }

    assert(result.maxVertices == kClusterMaxVertices);
    assert(result.maxTriangles == kClusterMaxTriangles);

    return result;
}

nvrhi::BufferHandle ClusterAccelBuilder::GenerateStructuredClusterTemplateArgs(const TemplateGrids &grids, nvrhi::ICommandList* commandList)
{
    // Align buffer size to 4 bytes for Vulkan vkCmdUpdateBuffer compatibility
    size_t indexDataSize = grids.indices.size() * sizeof(grids.indices[0]);
    nvrhi::BufferDesc indexBufferDesc = {
        .byteSize = alignBufferSize(indexDataSize),
        .debugName = "ClusterTemplateIndices",
        .structStride = sizeof(grids.indices[0]),
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };

    nvrhi::BufferHandle indexBuffer = CreateBuffer(indexBufferDesc, m_device.Get());
    if (grids.indices.size() > 0)
    {
        // writeBuffer uploads data to indexBuffer (use original data size, buffer is aligned)
        commandList->writeBuffer(indexBuffer, grids.indices.data(), indexDataSize);
    }
    // CRITICAL: Store in m_templateBuffers to keep alive - GPU addresses in cluster args reference this buffer
    m_templateBuffers.indexBuffer = indexBuffer;

    nvrhi::BufferDesc vertexBufferDesc = {
        .byteSize = grids.vertices.size() * sizeof(grids.vertices[0]),
        .debugName = "ClusterTemplateVertices",
        .format = nvrhi::Format::RGB32_FLOAT,
        .isVertexBuffer = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::AccelStructBuildInput,
        .keepInitialState = true,
    };
    nvrhi::BufferHandle vertexBuffer = CreateBuffer(vertexBufferDesc, m_device.Get());
    if (grids.vertices.size() > 0)
    {
        // writeBuffer uploads data to vertexBuffer
        commandList->writeBuffer(vertexBuffer, grids.vertices.data(), grids.vertices.size() * sizeof(grids.vertices[0]));
    }
    // CRITICAL: Store in m_templateBuffers to keep alive - GPU addresses in cluster args reference this buffer
    m_templateBuffers.vertexBuffer = vertexBuffer;

    nvrhi::GpuVirtualAddress indexBufferAddress = indexBuffer->getGpuVirtualAddress();
    nvrhi::GpuVirtualAddress vertexBufferAddress = vertexBuffer->getGpuVirtualAddress();

    RTXMG_LOG(str::format("RTX MegaGeo: Base indexBufferAddress=", std::hex, indexBufferAddress,
                             " vertexBufferAddress=", std::hex, vertexBufferAddress));
    if (indexBufferAddress == 0) {
        Logger::err("RTX MegaGeo: indexBufferAddress is NULL!");
    }
    if (vertexBufferAddress == 0) {
        Logger::err("RTX MegaGeo: vertexBufferAddress is NULL!");
    }
    RTXMG_LOG(str::format("RTX MegaGeo: indexBuffer desc: byteSize=", indexBuffer->getDesc().byteSize,
                             " structStride=", indexBuffer->getDesc().structStride,
                             " isAccelStructBuildInput=", indexBuffer->getDesc().isAccelStructBuildInput));
    RTXMG_LOG(str::format("RTX MegaGeo: vertexBuffer desc: byteSize=", vertexBuffer->getDesc().byteSize,
                             " structStride=", vertexBuffer->getDesc().structStride,
                             " isVertexBuffer=", vertexBuffer->getDesc().isVertexBuffer,
                             " isAccelStructBuildInput=", vertexBuffer->getDesc().isAccelStructBuildInput));

    uint32_t indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat32bit);
    switch (sizeof(TemplateGrids::IndexType))
    {
    case 1: indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat8bit); break;
    case 2: indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat16bit); break;
    case 4: indexFormat = static_cast<uint32_t>(cluster::OperationIndexFormat::IndexFormat32bit); break;
    default: assert(false);
    }

    // Use the correct NVRHI cluster::IndirectTriangleTemplateArgs structure
    // This matches the Vulkan VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV
    std::vector<cluster::IndirectTriangleTemplateArgs> createTemplateArgData(grids.descs.size());
    for (uint32_t i = 0; i < createTemplateArgData.size(); i++)
    {
        const TemplateGridDesc& grid = grids.descs[i];

        // Zero-initialize unused bit fields
        createTemplateArgData[i] = { };
        createTemplateArgData[i] = cluster::IndirectTriangleTemplateArgs
        {
            .clusterId = 0,
            .clusterFlags = 0,
            .triangleCount = grid.getNumTriangles(),
            .vertexCount = grid.getNumVerts(),
            .positionTruncateBitCount = 0,
            .indexFormat = indexFormat,
            .opacityMicromapIndexFormat = 0,
            .baseGeometryIndexAndFlags = 0,
            .indexBufferStride = static_cast<uint16_t>(sizeof(grids.indices[0])),
            .vertexBufferStride = static_cast<uint16_t>(sizeof(grids.vertices[0]) * 3),
            .geometryIndexAndFlagsBufferStride = 0,
            .opacityMicromapIndexBufferStride = 0,
            .indexBuffer = indexBufferAddress + grid.indexOffset,
            .vertexBuffer = vertexBufferAddress + grid.vertexOffset,
            .geometryIndexAndFlagsBuffer = 0,
            .opacityMicromapArray = 0,
            .opacityMicromapIndexBuffer = 0,
            .instantiationBoundingBoxLimit = 0
        };

        // DEBUG: Log first few indirect args
        if (i < 3) {
            RTXMG_LOG(str::format("RTX MegaGeo: Template[", i, "] triCount=", createTemplateArgData[i].triangleCount,
                                     " vertCount=", createTemplateArgData[i].vertexCount, " indexOffset=", grid.indexOffset,
                                     " vertexOffset=", grid.vertexOffset));
            RTXMG_LOG(str::format("RTX MegaGeo: Template[", i, "] indexBuffer=", createTemplateArgData[i].indexBuffer,
                                     " vertexBuffer=", createTemplateArgData[i].vertexBuffer));
        }
    }

    nvrhi::BufferDesc clusterTemplateArgsDesc =
    {
        .byteSize = createTemplateArgData.size() * sizeof(createTemplateArgData[0]),
        .debugName = "ClusterTemplateArgs",
        .structStride = sizeof(createTemplateArgData[0]),
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    return CreateAndUploadBuffer(createTemplateArgData, clusterTemplateArgsDesc, commandList);
}

void ClusterAccelBuilder::InitStructuredClusterTemplates(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList)
{
    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates called, maxGeometryCountPerMesh=", maxGeometryCountPerMesh));
    RTXMG_LOG(str::format("RTX MegaGeo: Current buffer state: indexBuffer=", (void*)m_templateBuffers.indexBuffer.Get(),
                             " vertexBuffer=", (void*)m_templateBuffers.vertexBuffer.Get(),
                             " dataBuffer=", (void*)m_templateBuffers.dataBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: Template settings: stored maxGeo=", m_templateBuffers.maxGeometryCountPerMesh,
                             " stored quantNBits=", m_templateBuffers.quantNBits,
                             " config quantNBits=", m_tessellatorConfig.quantNBits));

    // only initialize if maxGeometryCount or quantNBits changes
    if (m_templateBuffers.dataBuffer.Get() != 0 &&
        m_templateBuffers.maxGeometryCountPerMesh == maxGeometryCountPerMesh &&
        m_templateBuffers.quantNBits == m_tessellatorConfig.quantNBits) {
        RTXMG_LOG("RTX MegaGeo: InitStructuredClusterTemplates - early return, templates already initialized");
        return;
    }
    RTXMG_LOG("RTX MegaGeo: InitStructuredClusterTemplates - building new templates");

    nvrhi::utils::ScopedMarker marker(commandList, "InitStructuredClusterTemplates");
    m_templateBuffers.maxGeometryCountPerMesh = maxGeometryCountPerMesh;
    m_templateBuffers.quantNBits = m_tessellatorConfig.quantNBits;

    TemplateGrids grids = GenerateTemplateGrids();
    
    // First compute the size of each template so we can build the address buffer
    // this will also act as the settings for further operations below.
    cluster::OperationParams operationParams =
    {
        .maxArgCount = kNumTemplates,
        .type = cluster::OperationType::ClasBuildTemplates,
        .mode = cluster::OperationMode::GetSizes,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxGeometryIndex = maxGeometryCountPerMesh,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = grids.totalTriangles,
            .maxTotalVertexCount = grids.totalVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
        }
    };
    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(operationParams);
        
    nvrhi::BufferHandle clusterTemplateArgsBuffer = GenerateStructuredClusterTemplateArgs(grids, commandList);
    
    // CRITICAL: Use member variable to keep buffer alive - GPU caches address references
    m_templateBuffers.sizesBuffer.Create(kNumTemplates, "ClusterTemplateSizes", m_device.Get());

    cluster::OperationDesc templateGetSizesDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = clusterTemplateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = m_templateBuffers.sizesBuffer.Get(),
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(templateGetSizesDesc);

    // readback templateSizes
    std::vector<uint32_t> templateSizes = m_templateBuffers.sizesBuffer.Download(commandList);

    if (m_tessellatorConfig.enableLogging)
    {
        m_templateBuffers.sizesBuffer.Log(commandList);
    }

    size_t totalTemplateSize = 0;
    for (uint32_t i = 0; i < kNumTemplates; i++)
    {
        totalTemplateSize += templateSizes[i];
    }
    
    // Create template data buffer based off of totalSize of all templates
    nvrhi::BufferDesc destDataDesc = {
        .byteSize = totalTemplateSize,
        .debugName = "ClusterTemplateData",
        .canHaveUAVs = true,
        .isAccelStructStorage = true,
        .initialState = nvrhi::ResourceStates::AccelStructWrite,
        .keepInitialState = true,
    };
    m_templateBuffers.dataBuffer = CreateBuffer(destDataDesc, m_device.Get());

    // Explicit Destination mode, calculate the address offset for each template to get a tight fit
    operationParams.type = cluster::OperationType::ClasBuildTemplates;
    operationParams.mode = cluster::OperationMode::ExplicitDestinations;

    nvrhi::GpuVirtualAddress baseAddress = m_templateBuffers.dataBuffer->getGpuVirtualAddress();
    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates - dataBuffer baseAddress=", std::hex, baseAddress));
    if (baseAddress == 0) {
        Logger::err("RTX MegaGeo: InitStructuredClusterTemplates - dataBuffer baseAddress is NULL!");
    }

    std::vector<nvrhi::GpuVirtualAddress> addresses(kNumTemplates);
    totalTemplateSize = 0;
    for (size_t i = 0; i < addresses.size(); i++)
    {
        addresses[i] = baseAddress + totalTemplateSize;
        totalTemplateSize += templateSizes[i];
    }
    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates - computed ", addresses.size(), " template addresses"));
#if RTXMG_VERBOSE_LOGGING
    if (!addresses.empty()) {
        RTXMG_LOG(str::format("RTX MegaGeo: Template First address=", std::hex, addresses[0], " last=", addresses.back()));
    }
#endif

    m_templateBuffers.addressesBuffer.Create(kNumTemplates, "ClusterTemplateDestAddressData", m_device.Get());
    m_templateBuffers.addressesBuffer.Upload(addresses, commandList);
    m_templateBuffers.addresses = addresses; // Store CPU-side copy for FillInstantiateTemplateArgs
    m_templateBuffers.instantiationSizesBuffer.Create(kNumTemplates, "ClusterTemplateInstantiationSizes", m_device.Get());

    // Log all addresses before cluster template build
    RTXMG_LOG(str::format("RTX MegaGeo: Before createClusterTemplateDesc:"));
    RTXMG_LOG(str::format("  clusterTemplateArgsBuffer ptr=", (void*)clusterTemplateArgsBuffer.Get()));
    RTXMG_LOG(str::format("  clusterTemplateArgsBuffer addr=", std::hex, clusterTemplateArgsBuffer ? clusterTemplateArgsBuffer->getGpuVirtualAddress() : 0));
    RTXMG_LOG(str::format("  addressesBuffer ptr=", (void*)m_templateBuffers.addressesBuffer.Get()));
    RTXMG_LOG(str::format("  addressesBuffer addr=", std::hex, m_templateBuffers.addressesBuffer.GetGpuVirtualAddress()));

    cluster::OperationDesc createClusterTemplateDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = clusterTemplateArgsBuffer.Get(),
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = m_templateBuffers.addressesBuffer.Get(),
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = 0,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };
    RTXMG_LOG("RTX MegaGeo: Calling executeMultiIndirectClusterOperation for createClusterTemplateDesc");
    commandList->executeMultiIndirectClusterOperation(createClusterTemplateDesc);
    RTXMG_LOG("RTX MegaGeo: createClusterTemplateDesc complete");

    if (m_tessellatorConfig.enableLogging)
    {
        m_templateBuffers.addressesBuffer.Log(commandList);
    }

    // Create and fill out the instantiate args buffer from addressesBuffer
    // Align structStride to 16 bytes for Vulkan minStorageBufferOffsetAlignment
    uint32_t instantiateArgElementSize = sizeof(cluster::IndirectInstantiateTemplateArgs);
    uint32_t instantiateArgAlignedStride = (instantiateArgElementSize + 15) & ~15;
    nvrhi::BufferDesc instantiateTemplateArgsDesc =
    {
        .byteSize = instantiateArgAlignedStride * kNumTemplates,
        .debugName = "InstantiateTemplateArgs",
        .structStride = instantiateArgAlignedStride,
        .canHaveUAVs = true,
        .isDrawIndirectArgs = true,
        .isAccelStructBuildInput = true,
        .initialState = nvrhi::ResourceStates::IndirectArgument,
        .keepInitialState = true,
    };

    RTXMGBuffer<cluster::IndirectInstantiateTemplateArgs> instantiateTemplateArgsBuffer(instantiateTemplateArgsDesc, m_device.Get());
    FillInstantiateTemplateArgs(instantiateTemplateArgsBuffer, m_templateBuffers.addressesBuffer, kNumTemplates, commandList);

    if (m_tessellatorConfig.enableLogging)
    {
        instantiateTemplateArgsBuffer.Log(commandList, [](std::ostream& ss, auto e)
            {
                ss << "{ct: " << std::hex << e.clusterTemplate <<
                    " | vb: " << std::hex << e.vertexBuffer.startAddress << "}";
                return true;
            });
    }

    // Execute GetSizes mode to fill out destSizes
    operationParams.type = cluster::OperationType::ClasInstantiateTemplates;
    operationParams.mode = cluster::OperationMode::GetSizes;
    
    cluster::OperationDesc instantiateTemplateGetSizesDesc =
    {
        .params = operationParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgsBuffer = instantiateTemplateArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .outSizesBuffer = m_templateBuffers.instantiationSizesBuffer,
        .outSizesOffsetInBytes = 0
    };
    commandList->executeMultiIndirectClusterOperation(instantiateTemplateGetSizesDesc);

    m_templateBuffers.instantiationSizes = m_templateBuffers.instantiationSizesBuffer.Download(commandList);

    RTXMG_LOG(str::format("RTX MegaGeo: InitStructuredClusterTemplates - Download complete, size=",
        m_templateBuffers.instantiationSizes.size()));
    if (!m_templateBuffers.instantiationSizes.empty()) {
        RTXMG_LOG(str::format("RTX MegaGeo: First 3 instantiation sizes: [0]=", m_templateBuffers.instantiationSizes[0],
            " [1]=", m_templateBuffers.instantiationSizes.size() > 1 ? m_templateBuffers.instantiationSizes[1] : 0,
            " [2]=", m_templateBuffers.instantiationSizes.size() > 2 ? m_templateBuffers.instantiationSizes[2] : 0));
    }

    if (m_tessellatorConfig.enableLogging)
    {
        m_templateBuffers.instantiationSizesBuffer.Log(commandList, { .wrap = false });
    }
    RTXMG_LOG("RTX MegaGeo: InitStructuredClusterTemplates - complete");
}

void ClusterAccelBuilder::BuildStructuredCLASes(ClusterAccels& accels, uint32_t maxGeometryCountPerMesh,
    const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::BuildStructuredCLASes");

    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - m_maxClusters=", m_maxClusters));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasPtrsBuffer ptr=", (void*)accels.clasPtrsBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasBuffer ptr=", (void*)accels.clasBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - m_clasIndirectArgDataBuffer ptr=", (void*)m_clasIndirectArgDataBuffer.Get()));

    nvrhi::GpuVirtualAddress clasPtrsAddr = accels.clasPtrsBuffer.GetGpuVirtualAddress();
    nvrhi::GpuVirtualAddress clasBufferAddr = accels.clasBuffer.GetBuffer() ? accels.clasBuffer.GetBuffer()->getGpuVirtualAddress() : 0;
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasPtrsAddr=", std::hex, clasPtrsAddr));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - clasBufferAddr=", std::hex, clasBufferAddr));

    if (clasPtrsAddr == 0) Logger::err("RTX MegaGeo: BuildStructuredCLASes - clasPtrsAddr is NULL!");
    if (clasBufferAddr == 0) Logger::err("RTX MegaGeo: BuildStructuredCLASes - clasBufferAddr is NULL!");

    cluster::OperationParams instantiateClasParams =
    {
        .maxArgCount = m_maxClusters,
        .type = cluster::OperationType::ClasInstantiateTemplates,
        .mode = cluster::OperationMode::ExplicitDestinations,
        .flags = cluster::OperationFlags::None,
        .clas =
        {
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .maxGeometryIndex = maxGeometryCountPerMesh,
            .maxUniqueGeometryCount = 1,
            .maxTriangleCount = kClusterMaxTriangles,
            .maxVertexCount = kClusterMaxVertices,
            .maxTotalTriangleCount = m_maxClusters * kClusterMaxTriangles,
            .maxTotalVertexCount = m_maxVertices,
            .minPositionTruncateBitCount = m_tessellatorConfig.quantNBits,
        }
    };

    cluster::OperationSizeInfo sizeInfo = m_device->getClusterOperationSizeInfo(instantiateClasParams);
    RTXMG_LOG(str::format("RTX MegaGeo: BuildStructuredCLASes - scratchSizeInBytes=", sizeInfo.scratchSizeInBytes));

    cluster::OperationDesc instantiateClasDesc =
    {
        .params = instantiateClasParams,
        .scratchSizeInBytes = sizeInfo.scratchSizeInBytes,
        .inIndirectArgCountBuffer = m_tessellationCountersBuffer,
        .inIndirectArgCountOffsetInBytes = tessCounterRange.byteOffset + kClusterCountByteOffset,
        .inIndirectArgsBuffer = m_clasIndirectArgDataBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = accels.clasPtrsBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = nullptr,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = nullptr,
        .outAccelerationStructuresOffsetInBytes = 0
    };

    RTXMG_LOG("RTX MegaGeo: BuildStructuredCLASes - calling executeMultiIndirectClusterOperation");
    commandList->executeMultiIndirectClusterOperation(instantiateClasDesc);
    RTXMG_LOG("RTX MegaGeo: BuildStructuredCLASes - complete");
}

void ClusterAccelBuilder::FillInstanceClusters(const RTXMGScene& scene, ClusterAccels& accels, nvrhi::ICommandList* commandList)
{
    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - instances.size()=", instances.size(),
        " m_numInstances=", m_numInstances, " buffer size=", m_fillClustersDispatchIndirectBuffer.GetBytes()));

    nvrhi::utils::ScopedMarker marker(commandList, "FillInstanceClusters");
    stats::clusterAccelSamplers.fillClustersTime.Start(commandList);

#if RTXMG_CHRONO_TIMING
    auto fillStart = std::chrono::high_resolution_clock::now();
    float setupTimeMs = 0.0f;
    float bindingTimeMs = 0.0f;
    float dispatchTimeMs = 0.0f;
#endif

    uint32_t surfaceOffset{ 0 };
    // Limit loop to m_numInstances to avoid buffer overflow on indirect dispatch buffer
    uint32_t maxInstances = std::min(uint32_t(instances.size()), m_numInstances);
    if (instances.size() > m_numInstances) {
        dxvk::Logger::warn(dxvk::str::format("RTX MegaGeo: FillInstanceClusters - instances.size()=", instances.size(),
            " > m_numInstances=", m_numInstances, ", limiting to ", m_numInstances));
    }
    for (uint32_t instanceIndex = 0; instanceIndex < maxInstances; ++instanceIndex)
    {
#if RTXMG_CHRONO_TIMING
        auto instStart = std::chrono::high_resolution_clock::now();
#endif
        const auto& instance = instances[instanceIndex];

        // Bounds check to prevent crash - skip instances with invalid meshID
        if (instance.meshID >= subdMeshes.size()) {
            dxvk::Logger::warn(dxvk::str::format("RTX MegaGeo: FillInstanceClusters - meshID ", instance.meshID,
                " out of bounds (subdMeshes.size()=", subdMeshes.size(), "), skipping instance ", instanceIndex));
            continue;
        }

        assert(instance.meshInstance.get());
        const auto& donutMeshInfo = instance.meshInstance->GetMesh();
        assert(donutMeshInfo.get());
        uint32_t firstGeometryIndex = donutMeshInfo.geometries[0]->globalGeometryIndex;

        const auto& subd = *subdMeshes[instance.meshID];
        
        const uint32_t surfaceCount = subd.SurfaceCount();

        if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
            m_tessellatorConfig.debugClusterIndex >= 0 &&
            m_tessellatorConfig.debugLaneIndex >= 0)
        {
            commandList->clearBufferUInt(m_debugBuffer.Get(), 0);
        }

        FillClustersParams params = {};
        params.instanceIndex = instanceIndex;
        params.quantNBits = m_tessellatorConfig.quantNBits;
        params.isolationLevel = m_tessellatorConfig.isolationLevel;
        params.globalDisplacementScale = m_tessellatorConfig.displacementScale;
        params.clusterPattern = uint32_t(m_tessellatorConfig.clusterPattern);
        params.firstGeometryIndex = firstGeometryIndex;
        params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
        params.debugClusterIndex = uint32_t(m_tessellatorConfig.debugClusterIndex);
        params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
        commandList->writeBuffer(m_fillClustersParamsBuffer, &params, sizeof(FillClustersParams));

        // Bindings matching sample style (separate namespaces per resource type)
        // Order must match [[vk::binding]] in fill_clusters.comp.slang
        size_t gridSamplerStride = m_gridSamplersBuffer.GetElementBytes();
        auto bindingSetDesc = nvrhi::BindingSetDesc()
            // SRVs (0-19)
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_gridSamplersBuffer,
                nvrhi::Format::UNKNOWN,
                nvrhi::BufferRange(surfaceOffset * gridSamplerStride, surfaceCount * gridSamplerStride)))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_clusterOffsetCountsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_clustersBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subd.m_positionsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subd.m_vertexDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subd.m_vertexDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subd.m_vertexDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subd.GetTopologyMap()->plansBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subd.GetTopologyMap()->subpatchTreesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subd.GetTopologyMap()->patchPointIndicesArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subd.GetTopologyMap()->stencilMatrixArraysBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, subd.m_vertexDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, scene.GetGeometryBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, scene.GetMaterialBuffer()))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subd.m_surfaceToGeometryIndexBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subd.m_texcoordDeviceData.surfaceDescriptors))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subd.m_texcoordDeviceData.controlPointIndices))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(17, subd.m_texcoordDeviceData.patchPointsOffsets))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(18, subd.m_texcoordDeviceData.patchPoints))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(19, subd.m_texcoordsBuffer))
            // Sampler (20)
            .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))
            // UAVs (21-24)
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, accels.clusterVertexPositionsBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, accels.clusterShadingDataBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_debugBuffer))
            .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterVertexNormalsBuffer))
            // Constant buffer (25)
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_fillClustersParamsBuffer));

        // Create layout once, then reuse for all binding sets
        if (!m_fillClustersBL)
        {
            auto layoutDesc = nvrhi::BindingLayoutDesc()
                .setVisibility(nvrhi::ShaderType::Compute);
            // SRVs 0-19
            for (uint32_t i = 0; i < 20; ++i)
                layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(i));
            // Sampler 0
            layoutDesc.addItem(nvrhi::BindingLayoutItem::Sampler(0));
            // UAVs 0-3
            for (uint32_t i = 0; i < 4; ++i)
                layoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(i));
            // Constant buffer 0
            layoutDesc.addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0));
            m_fillClustersBL = m_device->createBindingLayout(layoutDesc);
        }

        nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_fillClustersBL);
        if (!bindingSet)
        {
            Logger::err("Failed to create binding set for fill_clusters.hlsl");
        }
#if RTXMG_CHRONO_TIMING
        auto afterBinding = std::chrono::high_resolution_clock::now();
        bindingTimeMs += std::chrono::duration_cast<std::chrono::microseconds>(afterBinding - instStart).count() * 0.001f;
#endif

        auto GetFillClustersPSO = [this](const FillClustersPermutation& shaderPermutation)
            {
                if (!m_fillClustersPSOs[shaderPermutation.index()])
                {
                    std::vector<donut::engine::ShaderMacro> fillClustersMacros;
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("VERTEX_NORMALS", shaderPermutation.isVertexNormalsEnabled() ? "1" : "0"));
                    fillClustersMacros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));
                    nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_clusters.hlsl", "FillClustersMain", &fillClustersMacros, nvrhi::ShaderType::Compute);

                    auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                        .setComputeShader(shader)
                        .addBindingLayout(m_fillClustersBL)
                        .addBindingLayout(m_bindlessBL);

                    m_fillClustersPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
                }
                return m_fillClustersPSOs[shaderPermutation.index()];
            };
        
        if (!m_fillClustersTexcoordsPSO)
        {
            nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/fill_clusters.hlsl", "FillClustersTexcoordsMain", nullptr, nvrhi::ShaderType::Compute);

            auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                .setComputeShader(shader)
                .addBindingLayout(m_fillClustersBL)
                .addBindingLayout(m_bindlessBL);

            m_fillClustersTexcoordsPSO = m_device->createComputePipeline(computePipelineDesc);
        }

        RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - instance ", instanceIndex, " creating compute state"));
        auto state = nvrhi::ComputeState()
            .addBindingSet(bindingSet)
            .addBindingSet(m_descriptorTable)  // Bindless descriptor table for displacement textures
            .setIndirectParams(m_fillClustersDispatchIndirectBuffer);

        // Pre-compute buffer bounds for all dispatch checks
        uint32_t fillBufferSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetBytes());
        uint32_t fillElementSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetElementBytes());

        if (m_tessellatorConfig.enableMonolithicClusterBuild)
        {
            RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - monolithic mode");
            FillClustersPermutation shaderPermutation = { subd.m_hasDisplacementMaterial, m_tessellatorConfig.enableVertexNormals, ShaderPermutationSurfaceType::All };
            state.setPipeline(GetFillClustersPSO(shaderPermutation));
            commandList->setComputeState(state);
            uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::Limit) * fillElementSize;
            RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - monolithic dispatchIndirect offset=", dispatchIndirectArgsOffset,
                " bufferSize=", fillBufferSize, " instanceIndex=", instanceIndex));
            if (dispatchIndirectArgsOffset + fillElementSize > fillBufferSize) {
                Logger::err(str::format("RTX MegaGeo: BUFFER OVERFLOW DETECTED! monolithic offset=", dispatchIndirectArgsOffset,
                    " + elementSize=", fillElementSize, " > bufferSize=", fillBufferSize, " instanceIndex=", instanceIndex));
            } else {
                commandList->dispatchIndirect(dispatchIndirectArgsOffset);
            }
        }
        else
        {
            RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - permutation mode");
            for (uint32_t i = 0; i <= uint32_t(ShaderPermutationSurfaceType::Limit); i++)
            {
                FillClustersPermutation shaderPermutation = { subd.m_hasDisplacementMaterial, m_tessellatorConfig.enableVertexNormals, ShaderPermutationSurfaceType(i) };
                RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - GetFillClustersPSO permutation ", i));
                state.setPipeline(GetFillClustersPSO(shaderPermutation));
                commandList->setComputeState(state);
                uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType(i)) * fillElementSize;
                RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - permutation dispatchIndirect i=", i, " offset=", dispatchIndirectArgsOffset));
                if (dispatchIndirectArgsOffset + fillElementSize > fillBufferSize) {
                    Logger::err(str::format("RTX MegaGeo: BUFFER OVERFLOW DETECTED! permutation i=", i, " offset=", dispatchIndirectArgsOffset,
                        " + elementSize=", fillElementSize, " > bufferSize=", fillBufferSize, " instanceIndex=", instanceIndex));
                } else {
                    commandList->dispatchIndirect(dispatchIndirectArgsOffset);
                }
            }
        }

        RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - texcoords dispatch");
        state.setPipeline(m_fillClustersTexcoordsPSO);
        commandList->setComputeState(state);
        uint32_t dispatchIndirectArgsOffset = (instanceIndex * ClusterDispatchType::NumTypes + ClusterDispatchType::AllTypes) * uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
        uint32_t bufferSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetBytes());
        uint32_t elementSize = static_cast<uint32_t>(m_fillClustersDispatchIndirectBuffer.GetElementBytes());
        RTXMG_LOG(str::format("RTX MegaGeo: FillInstanceClusters - texcoords dispatchIndirect offset=", dispatchIndirectArgsOffset,
            " bufferSize=", bufferSize, " instanceIndex=", instanceIndex, " m_numInstances=", m_numInstances));
        // Bounds check: ensure offset + element size <= buffer size
        if (dispatchIndirectArgsOffset + elementSize > bufferSize) {
            Logger::err(str::format("RTX MegaGeo: BUFFER OVERFLOW DETECTED! texcoords dispatchIndirect offset=", dispatchIndirectArgsOffset,
                " + elementSize=", elementSize, " = ", dispatchIndirectArgsOffset + elementSize,
                " > bufferSize=", bufferSize, " instanceIndex=", instanceIndex, " m_numInstances=", m_numInstances));
            continue; // Skip this dispatch to avoid crash
        }
        commandList->dispatchIndirect(dispatchIndirectArgsOffset);
        RTXMG_LOG("RTX MegaGeo: FillInstanceClusters - instance complete");

        surfaceOffset += surfaceCount;

        if (m_tessellatorConfig.debugSurfaceIndex >= 0 &&
            m_tessellatorConfig.debugClusterIndex >= 0 &&
            m_tessellatorConfig.debugLaneIndex >= 0)
        {
            Logger::info(str::format("Fill Clusters Debug Instance:", instanceIndex, " Mesh:", donutMeshInfo.name, " (Surface:", m_tessellatorConfig.debugSurfaceIndex,
                " Cluster:", m_tessellatorConfig.debugClusterIndex, " Lane:", m_tessellatorConfig.debugLaneIndex, ")"));

            auto debugOutput = m_debugBuffer.Download(commandList);
            uint numElements = debugOutput.front().payloadType;
            vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
        }
#if RTXMG_CHRONO_TIMING
        auto instEnd = std::chrono::high_resolution_clock::now();
        dispatchTimeMs += std::chrono::duration_cast<std::chrono::microseconds>(instEnd - afterBinding).count() * 0.001f;
#endif
    }

    stats::clusterAccelSamplers.fillClustersTime.Stop();
#if RTXMG_CHRONO_TIMING
    auto fillEnd = std::chrono::high_resolution_clock::now();
    float totalMs = std::chrono::duration_cast<std::chrono::microseconds>(fillEnd - fillStart).count() * 0.001f;
    Logger::info(str::format(">>> RTXMG CHRONO: FillInstanceClusters TOTAL=", totalMs, "ms binding=", bindingTimeMs, "ms dispatch=", dispatchTimeMs, "ms instances=", maxInstances));
#endif
}

void ClusterAccelBuilder::ComputeInstanceClusterTiling(ClusterAccels& accels,
    const RTXMGScene& scene,
    uint32_t instanceIndex,
    uint32_t surfaceOffset,
    uint32_t surfaceCount,
    const nvrhi::BufferRange& tessCounterRange,
    nvrhi::ICommandList* commandList)
{
    using namespace dxvk;
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling entry, instanceIndex=", instanceIndex, " surfaceOffset=", surfaceOffset, " surfaceCount=", surfaceCount));

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instance = scene.GetSubdMeshInstances()[instanceIndex];
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - got instance");

    const SubdivisionSurface& subdivisionSurface = *subdMeshes[instance.meshID];
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - got subdivisionSurface, meshID=", instance.meshID));

    assert(instance.meshInstance.get());
    const auto& donutMeshInfo = instance.meshInstance->GetMesh();
    assert(donutMeshInfo.get());
    uint32_t firstGeometryIndex = donutMeshInfo.geometries[0]->globalGeometryIndex;
    const auto& localToWorld = instance.localToWorld;
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - firstGeometryIndex=", firstGeometryIndex));

    // Only clear debug buffer when debugging is enabled (matching sample behavior)
    if (m_tessellatorConfig.debugSurfaceIndex >= 0 && m_tessellatorConfig.debugLaneIndex >= 0)
    {
        commandList->clearBufferUInt(m_debugBuffer.Get(), 0);
        RTXMG_LOG("RTX MegaGeo: Debug buffer cleared for debugging");
    }

    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating params");
    ComputeClusterTilingParams params = {};

    // Debug: Log struct layout - trace all fields to find alignment mismatch
#if RTXMG_VERBOSE_LOGGING
    RTXMG_LOG(str::format("RTX MegaGeo: STRUCT LAYOUT - sizeof=", sizeof(ComputeClusterTilingParams)));
    RTXMG_LOG(str::format("  offset(surfaceStart)=", offsetof(ComputeClusterTilingParams, surfaceStart),
        " offset(matWorldToClip)=", offsetof(ComputeClusterTilingParams, matWorldToClip),
        " offset(localToWorld)=", offsetof(ComputeClusterTilingParams, localToWorld)));
    RTXMG_LOG(str::format("  offset(cameraPos)=", offsetof(ComputeClusterTilingParams, cameraPos),
        " offset(aabb)=", offsetof(ComputeClusterTilingParams, aabb),
        " offset(edgeSegments)=", offsetof(ComputeClusterTilingParams, edgeSegments)));
    RTXMG_LOG(str::format("  offset(firstGeometryIndex)=", offsetof(ComputeClusterTilingParams, firstGeometryIndex),
        " offset(fineTessellationRate)=", offsetof(ComputeClusterTilingParams, fineTessellationRate),
        " offset(viewportSize)=", offsetof(ComputeClusterTilingParams, viewportSize)));
    RTXMG_LOG(str::format("  sizeof(float4)=", sizeof(float4),
        " sizeof(float3)=", sizeof(float3),
        " sizeof(Box3)=", sizeof(Box3),
        " sizeof(float4x4)=", sizeof(float4x4)));
#endif

    params.debugSurfaceIndex = uint32_t(m_tessellatorConfig.debugSurfaceIndex);
    params.debugLaneIndex = uint32_t(m_tessellatorConfig.debugLaneIndex);
    RTXMG_LOG(str::format("RTX MegaGeo: params - camera ptr=", (void*)m_tessellatorConfig.camera));

    // Convert dxvk matrices to float4x4
    RTXMG_LOG("RTX MegaGeo: params - getting projection matrix");
    auto projMatrix = m_tessellatorConfig.camera->GetProjectionMatrix();
    RTXMG_LOG("RTX MegaGeo: params - getting view matrix");
    auto viewMatrix = m_tessellatorConfig.camera->GetViewMatrix();
    RTXMG_LOG("RTX MegaGeo: params - multiplying matrices");
    auto viewProj = projMatrix * viewMatrix;
    RTXMG_LOG("RTX MegaGeo: params - copying matWorldToClip");
    memcpy(&params.matWorldToClip, &viewProj.data[0][0], sizeof(float) * 16);

    // Log viewProj matrix values
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row0=(", viewProj.data[0][0], ",", viewProj.data[0][1], ",", viewProj.data[0][2], ",", viewProj.data[0][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row1=(", viewProj.data[1][0], ",", viewProj.data[1][1], ",", viewProj.data[1][2], ",", viewProj.data[1][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row2=(", viewProj.data[2][0], ",", viewProj.data[2][1], ",", viewProj.data[2][2], ",", viewProj.data[2][3], ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: viewProj row3=(", viewProj.data[3][0], ",", viewProj.data[3][1], ",", viewProj.data[3][2], ",", viewProj.data[3][3], ")"));

    // DEBUG: Test project a sample point at (5, 5, 10) to see screen coordinates
#if RTXMG_VERBOSE_LOGGING
    {
        float testX = 5.0f, testY = 5.0f, testZ = 10.0f;
        float clipX = testX * viewProj.data[0][0] + testY * viewProj.data[1][0] + testZ * viewProj.data[2][0] + viewProj.data[3][0];
        float clipY = testX * viewProj.data[0][1] + testY * viewProj.data[1][1] + testZ * viewProj.data[2][1] + viewProj.data[3][1];
        float clipZ = testX * viewProj.data[0][2] + testY * viewProj.data[1][2] + testZ * viewProj.data[2][2] + viewProj.data[3][2];
        float clipW = testX * viewProj.data[0][3] + testY * viewProj.data[1][3] + testZ * viewProj.data[2][3] + viewProj.data[3][3];
        float ndcX = clipX / clipW;
        float ndcY = clipY / clipW;
        float screenX = (ndcX * 0.5f + 0.5f) * m_tessellatorConfig.viewportSize.x;
        float screenY = (ndcY * 0.5f + 0.5f) * m_tessellatorConfig.viewportSize.y;
        RTXMG_LOG(str::format("RTX MegaGeo: TEST POINT (5,5,10) -> clip=(", clipX, ",", clipY, ",", clipZ, ",", clipW,
            ") ndc=(", ndcX, ",", ndcY, ") screen=(", screenX, ",", screenY, ")"));
    }
#endif

    RTXMG_LOG("RTX MegaGeo: params - copying localToWorld");
    // Use column-major format matching sample's affineToColumnMajor (48 bytes = 3 x float4)
    // Row 0: (m00, m10, m20, tx) - column 0 of 3x3 + tx
    // Row 1: (m01, m11, m21, ty) - column 1 of 3x3 + ty
    // Row 2: (m02, m12, m22, tz) - column 2 of 3x3 + tz
    params.localToWorld[0] = float4(localToWorld.data[0][0], localToWorld.data[1][0], localToWorld.data[2][0], localToWorld.data[3][0]);
    params.localToWorld[1] = float4(localToWorld.data[0][1], localToWorld.data[1][1], localToWorld.data[2][1], localToWorld.data[3][1]);
    params.localToWorld[2] = float4(localToWorld.data[0][2], localToWorld.data[1][2], localToWorld.data[2][2], localToWorld.data[3][2]);

    // Log localToWorld values sent to shader
    RTXMG_LOG(str::format("RTX MegaGeo: localToWorld[0]=(", params.localToWorld[0].x, ",", params.localToWorld[0].y, ",", params.localToWorld[0].z, ",", params.localToWorld[0].w, ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: localToWorld[1]=(", params.localToWorld[1].x, ",", params.localToWorld[1].y, ",", params.localToWorld[1].z, ",", params.localToWorld[1].w, ")"));
    RTXMG_LOG(str::format("RTX MegaGeo: localToWorld[2]=(", params.localToWorld[2].x, ",", params.localToWorld[2].y, ",", params.localToWorld[2].z, ",", params.localToWorld[2].w, ")"));

    params.viewportSize.x = float(m_tessellatorConfig.viewportSize.x);
    params.viewportSize.y = float(m_tessellatorConfig.viewportSize.y);

    // Update stats renderSize for profiler display
    stats::clusterAccelSamplers.renderSize.x = static_cast<int>(m_tessellatorConfig.viewportSize.x);
    stats::clusterAccelSamplers.renderSize.y = static_cast<int>(m_tessellatorConfig.viewportSize.y);

    params.firstGeometryIndex = firstGeometryIndex;
    params.isolationLevel = m_tessellatorConfig.isolationLevel;
    params.coarseTessellationRate = m_tessellatorConfig.coarseTessellationRate;
    params.fineTessellationRate = m_tessellatorConfig.fineTessellationRate;
    RTXMG_LOG(str::format("RTX MegaGeo: tessRates - coarse=", params.coarseTessellationRate,
        " fine=", params.fineTessellationRate,
        " tessFactor=", params.coarseTessellationRate / params.fineTessellationRate));

    // Log key parameters once per frame (instance 0 only) for debugging screenspace tessellation
    if (instanceIndex == 0) {
        Logger::info(str::format(">>> RTXMG PARAMS: viewport=(", params.viewportSize.x, ",", params.viewportSize.y,
            ") tessRate=", params.fineTessellationRate, " isolation=", params.isolationLevel));
    }

    RTXMG_LOG("RTX MegaGeo: params - getting camera eye");

    // Convert dxvk Vector3 to float4 (w = padding, C++ float3 is 16 bytes breaking alignment)
    // NOTE: cameraPos is NOT used by SPHERICAL_PROJECTION anymore - we use clip.w instead
    auto eyePos = m_tessellatorConfig.camera->GetEye();
    params.cameraPos = float4(eyePos.x, eyePos.y, eyePos.z, 0.0f);
    RTXMG_LOG(str::format("RTX MegaGeo: cameraPos=(", params.cameraPos.x, ",", params.cameraPos.y, ",", params.cameraPos.z, ") [NOT USED - using clip.w]"));

    // Transform aabb from local space to world space using localToWorld matrix
    // This matches the sample's: params.aabb = subdivisionSurface.m_aabb * localToWorld;
    // Using the fast AABB transform algorithm from donut's box3::operator*
    {
        auto& aabb = subdivisionSurface.m_aabb;

        // Start with translation (DXVK Matrix4 is row-major, translation in row 3)
        float4 translation = float4(localToWorld.data[3][0], localToWorld.data[3][1], localToWorld.data[3][2], 0.0f);
        params.aabb.m_min = translation;
        params.aabb.m_max = translation;

        // Apply the linear transform (rotation + scale) to bounds
        // For each axis of the local AABB, compute contribution to world AABB
        for (int i = 0; i < 3; i++) {
            // Get row i of the linear part (columns 0-2 of Matrix4)
            float rowX = localToWorld.data[i][0];
            float rowY = localToWorld.data[i][1];
            float rowZ = localToWorld.data[i][2];

            // Scale by local min and max (aabb.m_mins is float[3])
            float minVal = aabb.m_mins[i];
            float maxVal = aabb.m_maxs[i];
            float4 e = float4(minVal * rowX, minVal * rowY, minVal * rowZ, 0.0f);
            float4 f = float4(maxVal * rowX, maxVal * rowY, maxVal * rowZ, 0.0f);

            // Accumulate min/max contributions
            params.aabb.m_min = float4(
                params.aabb.m_min.x + std::min(e.x, f.x),
                params.aabb.m_min.y + std::min(e.y, f.y),
                params.aabb.m_min.z + std::min(e.z, f.z), 0.0f);
            params.aabb.m_max = float4(
                params.aabb.m_max.x + std::max(e.x, f.x),
                params.aabb.m_max.y + std::max(e.y, f.y),
                params.aabb.m_max.z + std::max(e.z, f.z), 0.0f);
        }

        // Debug: Log transformed aabb values
#if RTXMG_VERBOSE_LOGGING
        float3 extent = float3(params.aabb.m_max.x - params.aabb.m_min.x,
                               params.aabb.m_max.y - params.aabb.m_min.y,
                               params.aabb.m_max.z - params.aabb.m_min.z);
        float diagonalLength = sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
        RTXMG_LOG(str::format("RTX MegaGeo: aabb (world space) min=(", params.aabb.m_min.x, ",", params.aabb.m_min.y, ",", params.aabb.m_min.z,
            ") max=(", params.aabb.m_max.x, ",", params.aabb.m_max.y, ",", params.aabb.m_max.z, ") diag=", diagonalLength));
#endif
    }

    params.enableBackfaceVisibility = m_tessellatorConfig.enableBackfaceVisibility;
    params.enableFrustumVisibility = m_tessellatorConfig.enableFrustumVisibility;
    params.enableHiZVisibility = m_tessellatorConfig.enableHiZVisibility && m_tessellatorConfig.zbuffer != nullptr;
    params.edgeSegments = m_tessellatorConfig.edgeSegments;
    params.globalDisplacementScale = m_tessellatorConfig.displacementScale;

    ONCE(Logger::info(str::format("RTX MegaGeo: Visibility params - frustum=", params.enableFrustumVisibility,
        " backface=", params.enableBackfaceVisibility, " HiZ=", params.enableHiZVisibility,
        " edgeSegments=(", params.edgeSegments.x, ",", params.edgeSegments.y, ",", params.edgeSegments.z, ",", params.edgeSegments.w, ")")));

    params.maxClasBlocks = uint32_t(m_maxClasBytes / size_t(cluster::kClasByteAlignment));
    params.maxClusters = m_maxClusters;
    params.maxVertices = m_maxVertices;
    params.clusterVertexPositionsBaseAddress = accels.clusterVertexPositionsBuffer.GetGpuVirtualAddress();
    params.clasDataBaseAddress = accels.clasBuffer.GetGpuVirtualAddress();

    // Safety check: if clasDataBaseAddress is 0, all CLAS addresses will be invalid
    if (params.clasDataBaseAddress == 0) {
        Logger::err("RTX MegaGeo: ComputeInstanceClusterTiling - clasDataBaseAddress is NULL! CLAS addresses will be invalid.");
    }
    if (params.clusterVertexPositionsBaseAddress == 0) {
        Logger::err("RTX MegaGeo: ComputeInstanceClusterTiling - clusterVertexPositionsBaseAddress is NULL!");
    }

    // Log addresses for debugging
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - clusterVertexPositionsBaseAddress=", std::hex, params.clusterVertexPositionsBaseAddress));
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - clasDataBaseAddress=", std::hex, params.clasDataBaseAddress));
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - maxClusters=", params.maxClusters, " maxVertices=", params.maxVertices, " maxClasBlocks=", params.maxClasBlocks));

    if (m_tessellatorConfig.zbuffer)
    {
        params.numHiZLODs = m_tessellatorConfig.zbuffer->GetNumHiZLODs();
        params.invHiZSize = m_tessellatorConfig.zbuffer->GetInvHiZSize();
    }
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - params filled, creating bindingSetDesc");

    // Create binding layouts - matching sample's 3 descriptor set structure:
    // Set 0: Main bindings (SRVs, UAVs, samplers, constant buffer)
    // Set 1: HiZ textures (space 1)
    // Set 2: Bindless textures
    if (!m_computeClusterTilingBL)
    {
        RTXMG_LOG("RTX MegaGeo: Creating main binding layout (set 0)");
        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.setVisibility(nvrhi::ShaderType::Compute)
            .setRegisterSpace(0)
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(5))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(6))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(7))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(8))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(9))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(10))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(11))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(12))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(13))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(14))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(15))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(16))
            .addItem(nvrhi::BindingLayoutItem::Sampler(0))
            .addItem(nvrhi::BindingLayoutItem::Sampler(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(0))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(5))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(6))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(7))
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(8));

        m_computeClusterTilingBL = m_device->createBindingLayout(layoutDesc);
        if (!m_computeClusterTilingBL)
        {
            Logger::err("Failed to create main binding layout for compute_cluster_tiling.hlsl");
        }
        RTXMG_LOG("RTX MegaGeo: Main binding layout created");
    }

    // Create HiZ binding layout (set 1) - shader expects HiZ at binding 0, set 1
    if (!m_computeClusterTilingHizBL)
    {
        RTXMG_LOG("RTX MegaGeo: Creating HiZ binding layout (set 1)");
        nvrhi::BindingLayoutDesc hizLayoutDesc;
        hizLayoutDesc.setVisibility(nvrhi::ShaderType::Compute)
            .setRegisterSpace(1)
            .setRegisterSpaceIsDescriptorSet(true)
            .addItem(nvrhi::BindingLayoutItem::Texture_SRV(0).setSize(HIZ_MAX_LODS));

        m_computeClusterTilingHizBL = m_device->createBindingLayout(hizLayoutDesc);
        if (!m_computeClusterTilingHizBL)
        {
            Logger::err("Failed to create HiZ binding layout for compute_cluster_tiling.hlsl");
        }
        RTXMG_LOG("RTX MegaGeo: HiZ binding layout created");

        // Create dummy HiZ binding set for when zbuffer is null
        nvrhi::BindingSetDesc dummyHizSetDesc;
        for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
        {
            dummyHizSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_dummyHiZTextures[i]).setArrayElement(i));
        }
        m_dummyHizBindingSet = m_device->createBindingSet(dummyHizSetDesc, m_computeClusterTilingHizBL);
        if (!m_dummyHizBindingSet)
        {
            Logger::err("Failed to create dummy HiZ binding set");
        }
        RTXMG_LOG("RTX MegaGeo: Dummy HiZ binding set created");
    }

    // Main binding set (set 0) - no HiZ textures, they're in set 1
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_computeClusterTilingParamsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, subdivisionSurface.m_positionsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, scene.GetGeometryBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(2, scene.GetMaterialBuffer()))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(3, subdivisionSurface.m_surfaceToGeometryIndexBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(4, subdivisionSurface.m_vertexDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(5, subdivisionSurface.m_vertexDeviceData.controlPointIndices));
    // DEBUG: Log surface descriptor buffer info
    RTXMG_LOG(str::format("RTX MegaGeo: Binding SurfaceDescriptors SRV(4) - buffer=",
        (void*)subdivisionSurface.m_vertexDeviceData.surfaceDescriptors.Get(),
        " bytes=", subdivisionSurface.m_vertexDeviceData.surfaceDescriptors ?
            subdivisionSurface.m_vertexDeviceData.surfaceDescriptors->getDesc().byteSize : 0,
        " surfaceCount=", surfaceCount));
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(6, subdivisionSurface.m_vertexDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(7, subdivisionSurface.GetTopologyMap()->plansBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(8, subdivisionSurface.GetTopologyMap()->subpatchTreesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(9, subdivisionSurface.GetTopologyMap()->patchPointIndicesArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(10, subdivisionSurface.GetTopologyMap()->stencilMatrixArraysBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(11, m_templateBuffers.instantiationSizesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(12, m_templateBuffers.addressesBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(13, subdivisionSurface.m_texcoordDeviceData.surfaceDescriptors))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(14, subdivisionSurface.m_texcoordDeviceData.controlPointIndices))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(15, subdivisionSurface.m_texcoordDeviceData.patchPointsOffsets))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(16, subdivisionSurface.m_texcoordsBuffer))
        .addItem(nvrhi::BindingSetItem::Sampler(0, scene.GetDisplacementSampler()))
        .addItem(nvrhi::BindingSetItem::Sampler(1, m_commonPasses->m_LinearClampSampler)) // hiZ sampler
        // UAV bindings
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_gridSamplersBuffer,
            nvrhi::Format::UNKNOWN,
            nvrhi::BufferRange(surfaceOffset * m_gridSamplersBuffer.GetElementBytes(), surfaceCount * m_gridSamplersBuffer.GetElementBytes())))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(1, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange));
    RTXMG_LOG(str::format("RTX MegaGeo: Binding tessCounters UAV(1) - range offset=", tessCounterRange.byteOffset,
                             " size=", tessCounterRange.byteSize, " buffer=", (void*)m_tessellationCountersBuffer.Get()));
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(2, m_clustersBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(3, accels.clusterShadingDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(4, m_clasIndirectArgDataBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(5, accels.clasPtrsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(6, subdivisionSurface.m_vertexDeviceData.patchPoints))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(7, subdivisionSurface.m_texcoordDeviceData.patchPoints))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(8, m_debugBuffer));
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - main bindingSetDesc built");

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_computeClusterTilingBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create main binding set for compute_cluster_tiling.hlsl");
    }
    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - main binding set created");

    // HiZ binding set (set 1) - use real zbuffer textures if available
    nvrhi::BindingSetHandle hizBindingSet;
    if (m_tessellatorConfig.zbuffer && m_tessellatorConfig.enableHiZVisibility)
    {
        RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating real HiZ binding set");

        // Initialize HiZ textures on first use by transitioning layout and clearing
        if (!m_hizInitialized)
        {
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - initializing HiZ textures (first use)");

            // Get underlying DxvkContext for layout transitions
            dxvk::NvrhiDxvkCommandList* dxvkCmdList = static_cast<dxvk::NvrhiDxvkCommandList*>(commandList);

            nvrhi::Color clearColor(std::numeric_limits<float>::max());

            // Clear each HiZ texture - this will transition from UNDEFINED layout
            for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
            {
                nvrhi::ITexture* hizTex = m_tessellatorConfig.zbuffer->GetHierarchyTexture(i);
                if (hizTex)
                {
                    RTXMG_LOG(str::format("RTX MegaGeo: Clearing HiZ texture level ", i));
                    commandList->clearTextureFloat(hizTex, nvrhi::AllSubresources, clearColor);
                }
            }
            m_hizInitialized = true;
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - HiZ textures initialized");
        }

        // Use cached binding set if zbuffer hasn't changed, otherwise create and cache new one
        if (m_cachedHizBuffer == m_tessellatorConfig.zbuffer && m_cachedHizBindingSet)
        {
            hizBindingSet = m_cachedHizBindingSet;
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - reusing cached HiZ binding set");
        }
        else
        {
            RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating and caching HiZ binding set");
            nvrhi::BindingSetDesc hizSetDesc;
            for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
            {
                nvrhi::ITexture* hizTex = m_tessellatorConfig.zbuffer->GetHierarchyTexture(i);
                hizSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, hizTex ? hizTex : m_dummyHiZTextures[i]).setArrayElement(i));
            }
            m_cachedHizBindingSet = m_device->createBindingSet(hizSetDesc, m_computeClusterTilingHizBL);
            m_cachedHizBuffer = m_tessellatorConfig.zbuffer;
            hizBindingSet = m_cachedHizBindingSet;
        }
    }
    else
    {
        RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - using dummy HiZ binding set");
        hizBindingSet = m_dummyHizBindingSet;
    }

    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating shaderPermutation");
    ComputeClusterTilingPermutation shaderPermutation(subdivisionSurface.m_hasDisplacementMaterial,
        m_tessellatorConfig.enableFrustumVisibility,
        m_tessellatorConfig.tessMode,
        m_tessellatorConfig.visMode,
        ShaderPermutationSurfaceType::PureBSpline);
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - shaderPermutation index=", shaderPermutation.index()));

    // Log tessellation mode once per frame (instance 0 only) for debugging screenspace approach
    if (instanceIndex == 0) {
        Logger::info(str::format("RTX MegaGeo: TessellationMode=", toString(shaderPermutation.tessellationMode()),
            " (SCREENSPACE: uses clip.w for LOD, NOT cameraPos)",
            " backface=", m_tessellatorConfig.enableBackfaceVisibility ? "YES (uses clip-space normal)" : "NO",
            " frustum=", m_tessellatorConfig.enableFrustumVisibility ? "YES" : "NO",
            " hiZ=", m_tessellatorConfig.enableHiZVisibility ? "YES" : "NO"));
    }

    auto GetComputeClusterTilingPSO = [this](const ComputeClusterTilingPermutation& shaderPermutation)
        {
            RTXMG_LOG(str::format("RTX MegaGeo: GetComputeClusterTilingPSO - index=", shaderPermutation.index()));
            if (!m_computeClusterTilingPSOs[shaderPermutation.index()])
            {
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - creating PSO");
                std::vector<donut::engine::ShaderMacro> macros;
                macros.push_back(donut::engine::ShaderMacro("DISPLACEMENT_MAPS", shaderPermutation.isDisplacementEnabled() ? "1" : "0"));
                macros.push_back(donut::engine::ShaderMacro("TESS_MODE", toString(shaderPermutation.tessellationMode())));
                macros.push_back(donut::engine::ShaderMacro("ENABLE_FRUSTUM_VISIBILITY", shaderPermutation.isFrustumVisibilityEnabled() ? "1" : "0"));
                macros.push_back(donut::engine::ShaderMacro("VIS_MODE", toString(shaderPermutation.visibilityMode())));
                macros.push_back(donut::engine::ShaderMacro("SURFACE_TYPE", toString(shaderPermutation.surfaceType())));
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - calling CreateShader");

                nvrhi::ShaderDesc tilingDesc(nvrhi::ShaderType::Compute);
                nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/compute_cluster_tiling.hlsl", "main", &macros, tilingDesc);
                RTXMG_LOG(str::format("RTX MegaGeo: GetComputeClusterTilingPSO - shader=", (void*)shader.Get()));

                // Store HiZ descriptor set layout in device for command list to use when binding set 1
                // This only needs to be done once (the layout is shared across all shader permutations)
                auto* nvrhiDevice = static_cast<NvrhiDxvkDevice*>(m_device.Get());
                if (nvrhiDevice && nvrhiDevice->getHiZDescriptorSetLayout() == VK_NULL_HANDLE) {
                    VkDescriptorSetLayout hiZLayout = m_shaderFactory.getHiZDescriptorSetLayout();
                    if (hiZLayout != VK_NULL_HANDLE) {
                        nvrhiDevice->setHiZDescriptorSetLayout(hiZLayout);
                        RTXMG_LOG("RTX MegaGeo: Stored HiZ descriptor set layout in device");
                    }
                }

                auto computePipelineDesc = nvrhi::ComputePipelineDesc()
                    .setComputeShader(shader)
                    .addBindingLayout(m_computeClusterTilingBL)      // Set 0: Main bindings
                    .addBindingLayout(m_computeClusterTilingHizBL)   // Set 1: HiZ textures
                    .addBindingLayout(m_bindlessBL);                 // Set 2: Bindless textures
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - creating pipeline");

                m_computeClusterTilingPSOs[shaderPermutation.index()] = m_device->createComputePipeline(computePipelineDesc);
                RTXMG_LOG("RTX MegaGeo: GetComputeClusterTilingPSO - pipeline created");
            }
            return m_computeClusterTilingPSOs[shaderPermutation.index()];
        };

    RTXMG_LOG("RTX MegaGeo: ComputeInstanceClusterTiling - creating compute state");
    auto state = nvrhi::ComputeState()
        .addBindingSet(bindingSet)           // Set 0: Main bindings
        .addBindingSet(hizBindingSet)        // Set 1: HiZ textures
        .addBindingSet(m_descriptorTable);   // Set 2: Bindless textures
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling - enableMonolithicClusterBuild=", m_tessellatorConfig.enableMonolithicClusterBuild));

    if (m_tessellatorConfig.enableMonolithicClusterBuild)
    {
        // Skip no limit surfaces
        params.surfaceStart = 0;
        params.surfaceEnd = subdivisionSurface.m_surfaceOffsets[uint32_t(SubdivisionSurface::SurfaceType::NoLimit)];
        uint32_t dispatchCount = params.surfaceEnd - params.surfaceStart;
        RTXMG_LOG(str::format("RTX MegaGeo: Monolithic - surfaceStart=", params.surfaceStart,
            " surfaceEnd=", params.surfaceEnd, " dispatchCount=", dispatchCount,
            " surfaceOffsets=[", subdivisionSurface.m_surfaceOffsets[0], ",",
            subdivisionSurface.m_surfaceOffsets[1], ",", subdivisionSurface.m_surfaceOffsets[2], ",",
            subdivisionSurface.m_surfaceOffsets[3], "]"));

        RTXMG_LOG("RTX MegaGeo: Monolithic - writeBuffer");
        commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));
        ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType::All;
        shaderPermutation.setSurfaceType(shaderSurfaceType);
        RTXMG_LOG("RTX MegaGeo: Monolithic - GetComputeClusterTilingPSO");
        state.setPipeline(GetComputeClusterTilingPSO(shaderPermutation));
        RTXMG_LOG("RTX MegaGeo: Monolithic - setComputeState");
        commandList->setComputeState(state);

        RTXMG_LOG("RTX MegaGeo: Monolithic - dispatch");
        commandList->dispatch(div_ceil(dispatchCount, kComputeClusterTilingWaves), 1, 1);
        RTXMG_LOG("RTX MegaGeo: Monolithic - dispatch complete");

        // Save cluster offset for this instance
        ClusterDispatchType dispatchType = ClusterDispatchType::AllTypes;
        RTXMG_LOG("RTX MegaGeo: Monolithic - CopyClusterOffset");
        CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
        RTXMG_LOG("RTX MegaGeo: Monolithic - CopyClusterOffset complete");
    }
    else
    {
        RTXMG_LOG("RTX MegaGeo: Loop mode - entering loop");
        // Loop
        for (uint32_t i = 0; i <= uint32_t(ClusterDispatchType::Limit); i++)
        {
            RTXMG_LOG(str::format("RTX MegaGeo: Loop iteration i=", i));
            SubdivisionSurface::SurfaceType subdSurfaceType = SubdivisionSurface::SurfaceType(i);

            // Skip no limit surfaces
            params.surfaceStart = subdivisionSurface.m_surfaceOffsets[uint32_t(subdSurfaceType)];
            params.surfaceEnd = subdivisionSurface.m_surfaceOffsets[uint32_t(subdSurfaceType) + 1];

            uint32_t dispatchCount = params.surfaceEnd - params.surfaceStart;
            RTXMG_LOG(str::format("RTX MegaGeo: Loop - surfaceStart=", params.surfaceStart, " surfaceEnd=", params.surfaceEnd, " dispatchCount=", dispatchCount));
            if (dispatchCount)
            {
                RTXMG_LOG("RTX MegaGeo: Loop - writeBuffer");
                commandList->writeBuffer(m_computeClusterTilingParamsBuffer, &params, sizeof(ComputeClusterTilingParams));

                ShaderPermutationSurfaceType shaderSurfaceType = ShaderPermutationSurfaceType(i);
                shaderPermutation.setSurfaceType(shaderSurfaceType);
                RTXMG_LOG("RTX MegaGeo: Loop - GetComputeClusterTilingPSO");
                state.setPipeline(GetComputeClusterTilingPSO(shaderPermutation));
                RTXMG_LOG("RTX MegaGeo: Loop - setComputeState");
                commandList->setComputeState(state);

                RTXMG_LOG("RTX MegaGeo: Loop - dispatch");
                commandList->dispatch(div_ceil(dispatchCount, kComputeClusterTilingWaves), 1, 1);
                RTXMG_LOG("RTX MegaGeo: Loop - dispatch complete");
            }
            // Save cluster offset for this instance
            ClusterDispatchType dispatchType = ClusterDispatchType(i);
            RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset");
            CopyClusterOffset(instanceIndex, dispatchType, tessCounterRange, commandList);
            RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset complete");
        }
        // Also copy to AllTypes slot so FillBlasFromClasArgs can read from it
        // (FillBlasFromClasArgs reads from ClusterDispatchType::All for all instances)
        RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset for AllTypes");
        CopyClusterOffset(instanceIndex, ClusterDispatchType::AllTypes, tessCounterRange, commandList);
        RTXMG_LOG("RTX MegaGeo: Loop - CopyClusterOffset for AllTypes complete");
        RTXMG_LOG("RTX MegaGeo: Loop mode - loop complete");
    }

    // Debug output download disabled for performance - enable ENABLE_SHADER_DEBUG to use
#if ENABLE_SHADER_DEBUG
    {
        Logger::info(str::format("RTX MegaGeo: Debug Instance:", instanceIndex, " Mesh:", donutMeshInfo.name));

        auto debugOutput = m_debugBuffer.Download(commandList);
        uint numElements = debugOutput.front().payloadType;
        Logger::info(str::format("RTX MegaGeo: Debug buffer numElements=", numElements));

        for (uint32_t i = 1; i <= std::min(numElements, 40u); ++i) {
            const auto& elem = debugOutput[i];
            if (elem.payloadType >= 9 && elem.payloadType <= 12) {
                Logger::info(str::format("RTX MegaGeo: Debug[", i, "] line=", elem.lineNumber,
                    " floats=[", elem.floatData.x, ",", elem.floatData.y, ",", elem.floatData.z, ",", elem.floatData.w, "]"));
            } else {
                Logger::info(str::format("RTX MegaGeo: Debug[", i, "] line=", elem.lineNumber,
                    " vals=[", elem.uintData.x, ",", elem.uintData.y, ",", elem.uintData.z, ",", elem.uintData.w, "]"));
            }
        }

        if (numElements > 0) {
            vectorlog::Log(debugOutput, ShaderDebugElement::OutputLambda, vectorlog::FormatOptions{ .wrap = false, .header = false, .elementIndex = false, .startIndex = 1, .count = numElements });
        }
    }
#endif
    RTXMG_LOG(str::format("RTX MegaGeo: ComputeInstanceClusterTiling complete for instance ", instanceIndex));

    // NOTE: Patchpoints logging removed - DownloadBuffer closes/reopens command list which
    // destroys bound HiZ image views and causes VK_ERROR_DEVICE_LOST in DXVK.
    // The sample code works differently because it uses a different nvrhi backend.
}

void ClusterAccelBuilder::CopyClusterOffset(uint32_t instanceIndex,
    ClusterDispatchType dispatchType, const nvrhi::BufferRange& tessCounterRange, nvrhi::ICommandList* commandList)
{
    // Bounds check: CopyClusterOffset shader writes to m_fillClustersDispatchIndirectBuffer at indices
    // based on instanceIndex. Prevent out-of-bounds writes by checking instanceIndex.
    if (instanceIndex >= m_numInstances) {
        Logger::err(str::format("RTX MegaGeo: CopyClusterOffset - instanceIndex ", instanceIndex,
            " >= m_numInstances ", m_numInstances, ", skipping to prevent buffer overflow"));
        return;
    }

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::CopyClusterOffset");
    CopyClusterOffsetParams params;
    params.instanceIndex = instanceIndex;
    params.dispatchTypeIndex = uint32_t(dispatchType);
    commandList->writeBuffer(m_copyClusterOffsetParamsBuffer, &params, sizeof(CopyClusterOffsetParams));

    // Use binding indices from copy_cluster_offset_binding_indices.h
    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(COPY_CLUSTER_OFFSET_TESS_COUNTERS_INPUT, m_tessellationCountersBuffer, nvrhi::Format::UNKNOWN, tessCounterRange))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_CLUSTER_OFFSET_COUNTS_OUTPUT, m_clusterOffsetCountsBuffer))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_FILL_INDIRECT_ARGS_OUTPUT, m_fillClustersDispatchIndirectBuffer))
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(COPY_CLUSTER_OFFSET_PARAMS, m_copyClusterOffsetParamsBuffer));

    // Create layout once, then reuse for all binding sets
    if (!m_copyClusterOffsetBL)
    {
        auto layoutDesc = nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::Compute)
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(COPY_CLUSTER_OFFSET_TESS_COUNTERS_INPUT))  // SRV t0
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_CLUSTER_OFFSET_COUNTS_OUTPUT))  // UAV u0
            .addItem(nvrhi::BindingLayoutItem::StructuredBuffer_UAV(COPY_CLUSTER_OFFSET_FILL_INDIRECT_ARGS_OUTPUT))  // UAV u1
            .addItem(nvrhi::BindingLayoutItem::ConstantBuffer(COPY_CLUSTER_OFFSET_PARAMS));  // CB b0
        m_copyClusterOffsetBL = m_device->createBindingLayout(layoutDesc);
    }

    nvrhi::BindingSetHandle bindingSet = m_device->createBindingSet(bindingSetDesc, m_copyClusterOffsetBL);
    if (!bindingSet)
    {
        Logger::err("Failed to create binding set for copy_cluster_offset shader");
    }

    if (!m_copyClusterOffsetPSO)
    {
        nvrhi::ShaderHandle shader = m_shaderFactory.CreateShader("cluster_builder/copy_cluster_offset.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);

        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(shader)
            .addBindingLayout(m_copyClusterOffsetBL);

        m_copyClusterOffsetPSO = m_device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_copyClusterOffsetPSO)
        .addBindingSet(bindingSet);
    commandList->setComputeState(state);
    commandList->dispatch(1, 1, 1);
}

void ClusterAccelBuilder::BuildBlasFromClas(ClusterAccels& accels, const Instance* instances, size_t instanceCount, nvrhi::ICommandList* commandList)
{
    //// Allocate and build BLASes
    nvrhi::utils::ScopedMarker marker(commandList, "Blas Build from Clas");
    stats::clusterAccelSamplers.buildBlasTime.Start(commandList);

    uint32_t numInstances = static_cast<uint32_t>(instanceCount);

    // Debug logging for NULL address detection
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - numInstances=", numInstances));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - clasPtrsBuffer ptr=", (void*)accels.clasPtrsBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasPtrsBuffer ptr=", (void*)accels.blasPtrsBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasSizesBuffer ptr=", (void*)accels.blasSizesBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasBuffer ptr=", (void*)accels.blasBuffer.Get()));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - m_blasFromClasIndirectArgsBuffer ptr=", (void*)m_blasFromClasIndirectArgsBuffer.Get()));

    nvrhi::GpuVirtualAddress clasPtrsBaseAddress = accels.clasPtrsBuffer.GetGpuVirtualAddress();
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - clasPtrsBaseAddress=", std::hex, clasPtrsBaseAddress));

    if (clasPtrsBaseAddress == 0) {
        Logger::err("RTX MegaGeo: BuildBlasFromClas - clasPtrsBaseAddress is NULL!");
    }

    FillBlasFromClasArgs(m_blasFromClasIndirectArgsBuffer, m_clusterOffsetCountsBuffer, clasPtrsBaseAddress, numInstances, commandList);

    if (m_tessellatorConfig.enableLogging)
    {
        m_blasFromClasIndirectArgsBuffer.Log(commandList, [](std::ostream& ss, const cluster::IndirectArgs& e)
            {
                ss << "{c: " << std::dec << e.clusterCount <<
                    " | addr: " << std::hex << e.clusterAddresses << "}";
                return true;
            });
    }

    // Check all addresses before the operation
    nvrhi::GpuVirtualAddress blasPtrsAddr = accels.blasPtrsBuffer.GetGpuVirtualAddress();
    nvrhi::GpuVirtualAddress blasBufferAddr = accels.blasBuffer.GetBuffer() ? accels.blasBuffer.GetBuffer()->getGpuVirtualAddress() : 0;

    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasPtrsAddr=", std::hex, blasPtrsAddr));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - blasBufferAddr=", std::hex, blasBufferAddr));
    RTXMG_LOG(str::format("RTX MegaGeo: BuildBlasFromClas - scratchSizeInBytes=", m_createBlasSizeInfo.scratchSizeInBytes));

    if (blasPtrsAddr == 0) Logger::err("RTX MegaGeo: BuildBlasFromClas - blasPtrsAddr is NULL!");
    if (blasBufferAddr == 0) Logger::err("RTX MegaGeo: BuildBlasFromClas - blasBufferAddr is NULL!");
    // Note: scratch buffer is now allocated on-demand by the nvrhi adapter if needed

    //// Build Operation
    cluster::OperationDesc createBlasDesc =
    {
        .params = m_createBlasParams,
        .scratchSizeInBytes = m_createBlasSizeInfo.scratchSizeInBytes,
        .inIndirectArgCountBuffer = nullptr,
        .inIndirectArgCountOffsetInBytes = 0,
        .inIndirectArgsBuffer = m_blasFromClasIndirectArgsBuffer,
        .inIndirectArgsOffsetInBytes = 0,
        .inOutAddressesBuffer = accels.blasPtrsBuffer,
        .inOutAddressesOffsetInBytes = 0,
        .outSizesBuffer = accels.blasSizesBuffer,
        .outSizesOffsetInBytes = 0,
        .outAccelerationStructuresBuffer = accels.blasBuffer,
        .outAccelerationStructuresOffsetInBytes = 0,
        // Note: outScratchBuffer not set - matching sample code behavior
    };

    RTXMG_LOG("RTX MegaGeo: BuildBlasFromClas - calling executeMultiIndirectClusterOperation");
    commandList->executeMultiIndirectClusterOperation(createBlasDesc);
    RTXMG_LOG("RTX MegaGeo: BuildBlasFromClas - executeMultiIndirectClusterOperation complete");

    stats::clusterAccelSamplers.buildBlasTime.Stop();
}
void ClusterAccelBuilder::UpdateMemoryAllocations(ClusterAccels& accels, uint32_t numInstances, uint32_t sceneSubdPatches)
{
    uint32_t maxClusters = std::min(kMaxApiClusterCount, m_tessellatorConfig.memorySettings.maxClusters);
    maxClusters = std::max(1u, maxClusters);

    // Reallocate memory if settings changed
    size_t maxClusterBlocks = (m_tessellatorConfig.memorySettings.clasBufferBytes + (size_t(cluster::kClasByteAlignment) - 1ull)) / size_t(cluster::kClasByteAlignment);
    maxClusterBlocks = std::max(1ull, maxClusterBlocks);
    size_t maxClasBytes = size_t(cluster::kClasByteAlignment) * maxClusterBlocks;

    // Calculate max vertices based on vertex buffer bytes (same for positions and normals since both are float3)
    uint32_t maxVertices = uint32_t(m_tessellatorConfig.memorySettings.vertexBufferBytes / sizeof(float3));
    maxVertices = std::max(kClusterMaxVertices, maxVertices);

    // Capture old values for logging
    uint32_t oldNumInstances = m_numInstances;
    uint32_t oldSceneSubdPatches = m_sceneSubdPatches;
    uint32_t oldMaxClusters = m_maxClusters;
    size_t oldMaxClasBytes = m_maxClasBytes;
    uint32_t oldMaxVertices = m_maxVertices;

    bool numInstancesChanged = m_numInstances != numInstances;
    bool sceneSubdPatchesChanged = m_sceneSubdPatches != sceneSubdPatches;
    bool numClustersChanged = m_maxClusters != maxClusters;
    bool clasBytesChanged = m_maxClasBytes != maxClasBytes;
    bool maxVerticesChanged = m_maxVertices != maxVertices;

    // Check if vertex normals setting changed by comparing current setting to buffer state
    bool prevVertexNormalsEnabled = accels.clusterVertexNormalsBuffer.GetBuffer() != nullptr && accels.clusterVertexNormalsBuffer.GetNumElements() == m_maxVertices;
    bool enableVertexNormalsChanged = (m_tessellatorConfig.enableVertexNormals != prevVertexNormalsEnabled);

    m_numInstances = numInstances;
    m_sceneSubdPatches = sceneSubdPatches;
    m_maxClusters = maxClusters;
    m_maxClasBytes = maxClasBytes;
    m_maxVertices = maxVertices;

    // No allocations needed
    if (!numInstancesChanged && !sceneSubdPatchesChanged && !numClustersChanged && !clasBytesChanged && !maxVerticesChanged && !enableVertexNormalsChanged)
    {
        return;
    }

    // Log which conditions triggered reallocation (helps debug frequent stalls)
    Logger::info(str::format("RTX MegaGeo: UpdateMemoryAllocations triggered - "
        "numInst=", numInstancesChanged, " subdPatches=", sceneSubdPatchesChanged,
        " clusters=", numClustersChanged, " clasBytes=", clasBytesChanged,
        " vertices=", maxVerticesChanged, " normals=", enableVertexNormalsChanged));
    if (numInstancesChanged)
        Logger::info(str::format("  numInstances: ", oldNumInstances, " -> ", numInstances));
    if (sceneSubdPatchesChanged)
        Logger::info(str::format("  sceneSubdPatches: ", oldSceneSubdPatches, " -> ", sceneSubdPatches));
    if (numClustersChanged)
        Logger::info(str::format("  maxClusters: ", oldMaxClusters, " -> ", maxClusters));
    if (clasBytesChanged)
        Logger::info(str::format("  maxClasBytes: ", oldMaxClasBytes, " -> ", maxClasBytes));
    if (maxVerticesChanged)
        Logger::info(str::format("  maxVertices: ", oldMaxVertices, " -> ", maxVertices));

    // Use deferred destruction instead of waitForIdle() to avoid GPU stalls
    // Old buffers are queued and destroyed after kDeferredDestructionFrames

    if (numInstancesChanged)
    {
        // Queue old buffers for deferred destruction (GPU may still be using them)
        if (m_copyClusterOffsetParamsBuffer || m_clusterOffsetCountsBuffer.GetBuffer())
        {
            DeferredBuffers deferred;
            deferred.frameQueued = m_currentFrameIndex;
            deferred.copyClusterOffsetParams = m_copyClusterOffsetParamsBuffer;
            deferred.clusterOffsetCounts = std::move(m_clusterOffsetCountsBuffer);
            deferred.fillClustersDispatchIndirect = std::move(m_fillClustersDispatchIndirectBuffer);
            deferred.blasFromClasIndirectArgs = std::move(m_blasFromClasIndirectArgsBuffer);
            deferred.blasPtrs = std::move(accels.blasPtrsBuffer);
            deferred.blasSizes = std::move(accels.blasSizesBuffer);
            m_deferredDestructionQueue.push_back(std::move(deferred));
        }

        // Clear handles (buffers are now owned by deferred queue)
        m_copyClusterOffsetParamsBuffer = nullptr;

        // Use a simple constant buffer instead of volatile - Vulkan has 64KB uniform buffer limit
        // Since we call writeBuffer before each dispatch, we don't need multiple versions
        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize = 256; // CopyClusterOffsetParams is 16 bytes, align to 256 for constant buffer
        cbDesc.debugName = "CopyClusterOffsetParams";
        cbDesc.isConstantBuffer = true;
        cbDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        cbDesc.keepInitialState = true;
        m_copyClusterOffsetParamsBuffer = m_device->createBuffer(cbDesc);

        m_clusterOffsetCountsBuffer.Create(m_numInstances * ClusterDispatchType::NumTypes, "ClusterOffsets", m_device.Get());
        nvrhi::BufferDesc dispatchIndirectDesc =
        {
            .byteSize = m_numInstances * ClusterDispatchType::NumTypes * m_fillClustersDispatchIndirectBuffer.GetElementBytes(),
            .debugName = "FillClustersIndirectArgs",
            .structStride = uint32_t(m_fillClustersDispatchIndirectBuffer.GetElementBytes()),
            .canHaveUAVs = true,
            .isDrawIndirectArgs = true,
            .initialState = nvrhi::ResourceStates::IndirectArgument,
            .keepInitialState = true,
        };
        m_fillClustersDispatchIndirectBuffer.Create(dispatchIndirectDesc, m_device.Get());

        // Create and fill out the instantiate args buffer from addressesBuffer
        // Align structStride to 16 bytes for Vulkan minStorageBufferOffsetAlignment
        uint32_t indirectArgElementSize = sizeof(cluster::IndirectArgs);
        uint32_t indirectArgAlignedStride = (indirectArgElementSize + 15) & ~15;
        nvrhi::BufferDesc clusterIndirectArgsDesc = {
            .byteSize = indirectArgAlignedStride * m_numInstances,
            .debugName = "cluster::IndirectArgs",
            .structStride = indirectArgAlignedStride,
            .canHaveUAVs = true,
            .isAccelStructBuildInput = true,
            .initialState = nvrhi::ResourceStates::ShaderResource,
            .keepInitialState = true,
        };
        m_blasFromClasIndirectArgsBuffer.Create(clusterIndirectArgsDesc, m_device.Get());
        accels.blasPtrsBuffer.Create(m_numInstances, "BlasPtrs", m_device.Get());
        accels.blasSizesBuffer.Create(m_numInstances, "BlasSizes", m_device.Get());
    }

    // For buffers without deferred destruction, we need waitForIdle before release
    // Instance buffers above use deferred destruction, but these don't yet
    bool needsWaitForIdle = sceneSubdPatchesChanged || numClustersChanged || clasBytesChanged || maxVerticesChanged || enableVertexNormalsChanged;
    if (needsWaitForIdle)
    {
        m_device->waitForIdle();
    }

    if (sceneSubdPatchesChanged)
    {
        m_gridSamplersBuffer.Release();
        m_gridSamplersBuffer.Create(m_sceneSubdPatches, "GridSamplers", m_device.Get());
    }

    if (numClustersChanged)
    {
        m_clustersBuffer.Release();
        m_clasIndirectArgDataBuffer.Release();
        accels.clusterShadingDataBuffer.Release();
        accels.clasPtrsBuffer.Release();

        m_clustersBuffer.Create(m_maxClusters, "clusters", m_device.Get());
        m_clasIndirectArgDataBuffer.Create(m_maxClusters, "indirect arg data", m_device.Get());

        accels.clusterShadingDataBuffer.Create(m_maxClusters, "cluster shading data", m_device.Get());
        accels.clasPtrsBuffer.Create(m_maxClusters, "ClasAddresses", m_device.Get());
    }

    RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - DEBUG: numClustersChanged=", numClustersChanged, " numInstancesChanged=", numInstancesChanged));

    if (numClustersChanged || numInstancesChanged)
    {
        RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - creating BLAS buffers, m_numInstances=", m_numInstances, " m_maxClusters=", m_maxClusters));
        accels.blasBuffer.Release();

        m_createBlasParams =
        {
            .maxArgCount = m_numInstances,
            .type = cluster::OperationType::BlasBuild,
            .mode = cluster::OperationMode::ImplicitDestinations,
            .flags = cluster::OperationFlags::None,
            .blas =
            {
                .maxClasPerBlasCount = m_maxClusters,
                .maxTotalClasCount = m_maxClusters
            }
        };
        m_createBlasSizeInfo = m_device->getClusterOperationSizeInfo(m_createBlasParams);
        RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - BLAS sizeInfo: resultMaxSizeInBytes=", m_createBlasSizeInfo.resultMaxSizeInBytes,
            " scratchSizeInBytes=", m_createBlasSizeInfo.scratchSizeInBytes));

        if (m_createBlasSizeInfo.resultMaxSizeInBytes == 0) {
            Logger::err("RTX MegaGeo: UpdateMemoryAllocations - resultMaxSizeInBytes is 0!");
        }
        if (m_createBlasSizeInfo.scratchSizeInBytes == 0) {
            Logger::warn("RTX MegaGeo: UpdateMemoryAllocations - scratchSizeInBytes is 0 (may be OK if no scratch needed)");
        }

        nvrhi::BufferDesc blasBufferDesc = {
            .byteSize = m_createBlasSizeInfo.resultMaxSizeInBytes,
            .debugName = "Blas Data",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        if (m_createBlasSizeInfo.resultMaxSizeInBytes > 0)
        {
            accels.blasBuffer.Create(blasBufferDesc, m_device.Get());
            RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - blasBuffer created, ptr=", (void*)accels.blasBuffer.GetBuffer().Get()));
            if (!accels.blasBuffer.GetBuffer())
            {
                Logger::err("RTX MegaGeo: UpdateMemoryAllocations - blasBuffer creation FAILED!");
            }
        }
        else
        {
            Logger::err("RTX MegaGeo: UpdateMemoryAllocations - cannot create blasBuffer with size 0!");
        }
        // Note: scratch buffer is allocated on-demand by nvrhi adapter (matching sample behavior)
    }
    else
    {
        RTXMG_LOG(str::format("RTX MegaGeo: UpdateMemoryAllocations - skipping BLAS buffer creation (no change)"));
    }

    if (clasBytesChanged)
    {
        accels.clasBuffer.Release();

        nvrhi::BufferDesc clasDataDesc =
        {
            .byteSize = m_maxClasBytes,
            .debugName = "ClasData",
            .canHaveUAVs = true,
            .isAccelStructStorage = true,
            .initialState = nvrhi::ResourceStates::AccelStructWrite,
            .keepInitialState = true,
        };
        accels.clasBuffer.Create(clasDataDesc, m_device.Get());
    }

    if (maxVerticesChanged)
    {
        accels.clusterVertexPositionsBuffer.Release();
        accels.clusterVertexPositionsBuffer.Create(m_maxVertices, "cluster vertex positions", m_device.Get());
    }
        
    if (maxVerticesChanged || enableVertexNormalsChanged)
    {
        accels.clusterVertexNormalsBuffer.Release();
        accels.clusterVertexNormalsBuffer.Create(m_tessellatorConfig.enableVertexNormals ? m_maxVertices : 1, "cluster vertex normals", m_device.Get());
    }
}

void ClusterAccelBuilder::EnsureTemplatesInitialized(uint32_t maxGeometryCountPerMesh, nvrhi::ICommandList* commandList)
{
    // Initialize cluster templates early, before any image views are bound
    // The sync Downloads in InitStructuredClusterTemplates close/reopen the command list
    // which destroys any bound resources (like HiZ textures)
    InitStructuredClusterTemplates(maxGeometryCountPerMesh, commandList);
}

void ClusterAccelBuilder::BuildAccel(const RTXMGScene& scene, const TessellatorConfig& config,
    ClusterAccels& accels, ClusterStatistics& stats, uint32_t frameIndex, nvrhi::ICommandList* commandList)
{
    // Process deferred destruction at start of frame - clean up old buffers from N frames ago
    // This avoids waitForIdle() during buffer reallocation which kills performance
    ProcessDeferredDestruction(frameIndex);
    m_currentFrameIndex = frameIndex;

#if RTXMG_CHRONO_TIMING
    auto chronoStart = std::chrono::high_resolution_clock::now();
    auto chronoSectionStart = chronoStart;
#endif
    m_tessellatorConfig = config;

    const auto& subdMeshes = scene.GetSubdMeshes();
    const auto& instances = scene.GetSubdMeshInstances();

    if (subdMeshes.empty() || instances.empty())
        return;

    uint32_t totalSubdPatches = scene.TotalSubdPatchCount();
    RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - instances=", instances.size(),
        " subdMeshes=", subdMeshes.size(), " totalSubdPatches=", totalSubdPatches));
#if RTXMG_CHRONO_TIMING
    auto setupStart = std::chrono::high_resolution_clock::now();
#endif
    UpdateMemoryAllocations(accels, uint32_t(instances.size()), totalSubdPatches);
#if RTXMG_CHRONO_TIMING
    auto afterMemAlloc = std::chrono::high_resolution_clock::now();
    float memAllocMs = std::chrono::duration_cast<std::chrono::microseconds>(afterMemAlloc - setupStart).count() * 0.001f;
    if (memAllocMs > 1.0f) {
        Logger::info(str::format(">>> RTXMG CHRONO: UpdateMemoryAllocations=", memAllocMs, "ms (SLOW - likely waitForIdle)"));
    }
#endif
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after UpdateMemoryAllocations");

    const uint32_t maxGeometryCountPerMesh = uint32_t(scene.GetSceneGraph()->GetMaxGeometryCountPerMesh());
    InitStructuredClusterTemplates(maxGeometryCountPerMesh, commandList);
#if RTXMG_CHRONO_TIMING
    auto afterTemplates = std::chrono::high_resolution_clock::now();
    float templatesMs = std::chrono::duration_cast<std::chrono::microseconds>(afterTemplates - afterMemAlloc).count() * 0.001f;
    if (templatesMs > 1.0f) {
        Logger::info(str::format(">>> RTXMG CHRONO: InitStructuredClusterTemplates=", templatesMs, "ms (SLOW)"));
    }
#endif
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after InitStructuredClusterTemplates");

    nvrhi::utils::ScopedMarker marker(commandList, "ClusterAccelBuilder::BuildAccel");
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after ScopedMarker");

    uint32_t tessCounterIndex = (m_buildAccelFrameIndex % kFrameCount);
    nvrhi::BufferRange tessCounterRange = { m_tessellationCountersBuffer.GetElementBytes() * tessCounterIndex, m_tessellationCountersBuffer.GetElementBytes() };
    RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - tessCounterIndex=", tessCounterIndex));
    RTXMG_LOG(str::format("RTX MegaGeo: tessCounterRange offset=", tessCounterRange.byteOffset,
                             " size=", tessCounterRange.byteSize,
                             " bufferSize=", m_tessellationCountersBuffer.GetBytes(),
                             " elementSize=", m_tessellationCountersBuffer.GetElementBytes()));

    // Clear tessellation counters for this frame
    TessellationCounters tessCounters = {};
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before UploadElement");
    m_tessellationCountersBuffer.UploadElement(tessCounters, tessCounterIndex, commandList);
    RTXMG_LOG(str::format("RTX MegaGeo: Uploaded zeroed counters to index ", tessCounterIndex,
                             " (clusters=", tessCounters.clusters, " desired=", tessCounters.desiredClusters, ")"));
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after UploadElement");

#if RTXMG_CHRONO_TIMING
    auto beforeClears = std::chrono::high_resolution_clock::now();
#endif
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before clearBufferUInt 1");
    commandList->clearBufferUInt(m_clusterOffsetCountsBuffer.Get(), 0);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before clearBufferUInt 2");
    commandList->clearBufferUInt(m_fillClustersDispatchIndirectBuffer.Get(), 0);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - before clearBufferUInt 3");
    // Clear BLAS indirect args to ensure any unprocessed instances have clusterCount = 0
    commandList->clearBufferUInt(m_blasFromClasIndirectArgsBuffer.Get(), 0);
#if RTXMG_CHRONO_TIMING
    auto afterClears = std::chrono::high_resolution_clock::now();
    float clearsMs = std::chrono::duration_cast<std::chrono::microseconds>(afterClears - beforeClears).count() * 0.001f;
    if (clearsMs > 1.0f) {
        Logger::info(str::format(">>> RTXMG CHRONO: BufferClears=", clearsMs, "ms (SLOW)"));
    }
#endif
    RTXMG_LOG("RTX MegaGeo: BuildAccel - after clearBufferUInt");

    // Transition dummy HiZ textures to ShaderResource on first use only
    // Use RtxContext's initImage to initialize the textures from UNDEFINED to their stable layout
    if (!m_tessellatorConfig.zbuffer && !m_dummyHiZTexturesInitialized)
    {
        RTXMG_LOG("RTX MegaGeo: Initializing dummy HiZ textures via initImage");

        VkImageSubresourceRange subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, VK_REMAINING_MIP_LEVELS,
            0, VK_REMAINING_ARRAY_LAYERS
        };

        for (uint32_t i = 0; i < HIZ_MAX_LODS; ++i)
        {
            if (m_dummyHiZTextures[i])
            {
                // Get the underlying DxvkImage from our NVRHI texture
                NvrhiDxvkTexture* nvrhiTexture = static_cast<NvrhiDxvkTexture*>(m_dummyHiZTextures[i].Get());
                const Rc<DxvkImage>& dxvkImage = nvrhiTexture->getDxvkImage();

                // Use initImage to transition from UNDEFINED to the image's stable layout
                // This is DXVK's standard way to initialize newly created images
                m_rtxContext->initImage(
                    dxvkImage,
                    subresourceRange,
                    VK_IMAGE_LAYOUT_UNDEFINED);

                RTXMG_LOG(str::format("RTX MegaGeo: Initialized DummyHiZ_Level_", i, " via initImage"));
            }
        }

        // Force the command list to be flushed so the init barriers are executed
        // This ensures the images are in the correct layout before any subsequent use
        m_rtxContext->flushCommandList();
        RTXMG_LOG("RTX MegaGeo: Flushed command list after dummy HiZ initialization");

        m_dummyHiZTexturesInitialized = true;
        RTXMG_LOG("RTX MegaGeo: Dummy HiZ texture initialization complete");
    }

    {
        RTXMG_LOG("RTX MegaGeo: BuildAccel - entering ComputeClusterTiling");
        nvrhi::utils::ScopedMarker marker(commandList, "ComputeClusterTiling");
        stats::clusterAccelSamplers.clusterTilingTime.Start(commandList);
        uint32_t surfaceOffset = 0;
        // Limit to m_numInstances to avoid buffer overflows
        uint32_t maxInstances = std::min(uint32_t(instances.size()), m_numInstances);
        RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - ComputeClusterTiling loop, maxInstances=", maxInstances));
#if RTXMG_CHRONO_TIMING
        float totalInstanceMs = 0.0f;
        uint32_t totalSurfaces = 0;
#endif
        for (uint32_t i = 0; i < maxInstances; ++i)
        {
#if RTXMG_CHRONO_TIMING
            auto instanceStart = std::chrono::high_resolution_clock::now();
#endif
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - loop iteration start i=", i));
            const auto& inst = instances[i];
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - got inst, meshID=", inst.meshID, " subdMeshes.size()=", subdMeshes.size()));

            // Bounds check to prevent crash
            if (inst.meshID >= subdMeshes.size()) {
                Logger::err(str::format("RTX MegaGeo: BuildAccel - meshID ", inst.meshID, " out of bounds (subdMeshes.size()=", subdMeshes.size(), "), skipping instance ", i));
                continue;
            }

            const auto& subd = *subdMeshes[inst.meshID];
            RTXMG_LOG("RTX MegaGeo: BuildAccel - got subd");

            uint32_t surfaceCount{ subd.SurfaceCount() };
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - instance ", i, " surfaceCount=", surfaceCount));


            ComputeInstanceClusterTiling(accels, scene, i, surfaceOffset, surfaceCount, tessCounterRange, commandList);
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - instance ", i, " ComputeInstanceClusterTiling complete"));

            surfaceOffset += surfaceCount;
            RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel - loop iteration end i=", i, " surfaceOffset=", surfaceOffset));
#if RTXMG_CHRONO_TIMING
            auto instanceEnd = std::chrono::high_resolution_clock::now();
            float instanceMs = std::chrono::duration_cast<std::chrono::microseconds>(instanceEnd - instanceStart).count() * 0.001f;
            totalInstanceMs += instanceMs;
            totalSurfaces += surfaceCount;
            // Log every 10th instance or if it took >5ms
            if (i % 10 == 0 || instanceMs > 5.0f) {
                Logger::info(str::format(">>> RTXMG CHRONO: Instance[", i, "] surfaces=", surfaceCount, " time=", instanceMs, "ms"));
            }
#endif
        }
        RTXMG_LOG("RTX MegaGeo: BuildAccel - ComputeClusterTiling loop complete");
        stats::clusterAccelSamplers.clusterTilingTime.Stop();
#if RTXMG_CHRONO_TIMING
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float tilingMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        float avgPerInstance = maxInstances > 0 ? totalInstanceMs / maxInstances : 0.0f;
        float avgPerSurface = totalSurfaces > 0 ? totalInstanceMs / totalSurfaces : 0.0f;
        Logger::info(str::format(">>> RTXMG CHRONO: ComputeClusterTiling TOTAL=", tilingMs, "ms instances=", maxInstances,
            " surfaces=", totalSurfaces, " avgPerInst=", avgPerInstance, "ms avgPerSurf=", avgPerSurface, "ms"));
        chronoSectionStart = chronoNow;
#endif
    }

    // NOTE: enableLogging block removed - Log()/Download() calls close/reopen command list
    // which destroys bound image views and causes VK_ERROR_DEVICE_LOST in DXVK.

    RTXMG_LOG("RTX MegaGeo: BuildAccel - calling FillInstanceClusters");
    FillInstanceClusters(scene, accels, commandList);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - FillInstanceClusters complete");
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float fillMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        Logger::info(str::format(">>> RTXMG CHRONO: FillInstanceClusters=", fillMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // Build CLASes for all instances at once
    RTXMG_LOG("RTX MegaGeo: BuildAccel - calling BuildStructuredCLASes");
    stats::clusterAccelSamplers.buildClasTime.Start(commandList);
    BuildStructuredCLASes(accels, maxGeometryCountPerMesh, tessCounterRange, commandList);
    stats::clusterAccelSamplers.buildClasTime.Stop();
    RTXMG_LOG("RTX MegaGeo: BuildAccel - BuildStructuredCLASes complete");
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float clasMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        Logger::info(str::format(">>> RTXMG CHRONO: BuildStructuredCLASes=", clasMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // Build BLAS unconditionally (matching sample behavior)
    // NOTE: Removed sync Download check for clusters > 0 because Download closes/reopens
    // command list which destroys bound image views in DXVK
    RTXMG_LOG("RTX MegaGeo: BuildAccel - calling BuildBlasFromClas");
    uint32_t blasInstanceCount = std::min(uint32_t(instances.size()), m_numInstances);
    BuildBlasFromClas(accels, instances.data(), blasInstanceCount, commandList);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - BuildBlasFromClas complete");
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float blasMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        Logger::info(str::format(">>> RTXMG CHRONO: BuildBlasFromClas=", blasMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // Async read of counters (reading previous frame's results for double-buffering)
    // On early frames, previous frame data may not exist yet, so fall back to current frame's data
    uint32_t readIndex = (tessCounterIndex + 1) % kFrameCount;
    RTXMG_LOG(str::format("RTX MegaGeo: About to download counters - writeIndex=", tessCounterIndex,
                             " readIndex=", readIndex, " frame=", m_buildAccelFrameIndex));
    RTXMG_LOG("RTX MegaGeo: BuildAccel - downloading counters");
    auto counterBufferData = m_tessellationCountersBuffer.Download(commandList, true);
    RTXMG_LOG("RTX MegaGeo: BuildAccel - counters downloaded");
#if RTXMG_CHRONO_TIMING
    {
        auto chronoNow = std::chrono::high_resolution_clock::now();
        float downloadMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoNow - chronoSectionStart).count() * 0.001f;
        Logger::info(str::format(">>> RTXMG CHRONO: DownloadCounters=", downloadMs, "ms"));
        chronoSectionStart = chronoNow;
    }
#endif

    // Log ALL counter indices to see which ones have data
#if RTXMG_VERBOSE_LOGGING
    for (uint32_t i = 0; i < kFrameCount; ++i) {
        RTXMG_LOG(str::format("RTX MegaGeo: Counter[", i, "] clusters=", counterBufferData[i].clusters,
                                 " desiredClusters=", counterBufferData[i].desiredClusters,
                                 " desiredTriangles=", counterBufferData[i].desiredTriangles,
                                 " desiredVertices=", counterBufferData[i].desiredVertices));
    }
#endif

    // If readIndex has no valid data (0 clusters), find the first index with valid data
    // This handles early frames where previous frame data doesn't exist yet
    TessellationCounters counters = counterBufferData[readIndex];
    if (counters.clusters == 0) {
        // Search for any index with valid cluster data, prefer current frame's index
        for (uint32_t i = 0; i < kFrameCount; ++i) {
            uint32_t checkIndex = (tessCounterIndex + kFrameCount - i) % kFrameCount;
            if (counterBufferData[checkIndex].clusters > 0) {
                readIndex = checkIndex;
                counters = counterBufferData[readIndex];
                RTXMG_LOG(str::format("RTX MegaGeo: Fallback to counter index ", readIndex, " with ", counters.clusters, " clusters"));
                break;
            }
        }
    }
    RTXMG_LOG(str::format("RTX MegaGeo: Using counters from index ", readIndex,
                             ": clusters=", counters.clusters, " desired=", counters.desiredClusters));

    // Record the desired required memory instead of the max
    stats.desired.m_numTriangles = counters.desiredTriangles;
    stats.desired.m_numClusters = counters.desiredClusters;
    stats.desired.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetElementBytes() * counters.desiredVertices;
    stats.desired.m_vertexNormalsBufferSize = m_tessellatorConfig.enableVertexNormals ? 
        (accels.clusterVertexNormalsBuffer.GetElementBytes() * counters.desiredVertices) : 0;
    stats.desired.m_clasSize = counters.DesiredClasBytes();
    stats.desired.m_clusterDataSize = (m_clustersBuffer.GetElementBytes() + 
        accels.clusterShadingDataBuffer.GetElementBytes() +
        accels.clasPtrsBuffer.GetElementBytes()) * counters.desiredClusters;
    stats.desired.m_blasSize = m_createBlasSizeInfo.resultMaxSizeInBytes;
    stats.desired.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    // Atomics are expensive so we don't track the number of allocated triangles
    stats.allocated.m_numTriangles = counters.desiredTriangles;
    stats.allocated.m_numClusters = m_maxClusters;
    stats.allocated.m_vertexBufferSize = accels.clusterVertexPositionsBuffer.GetBytes();
    stats.allocated.m_vertexNormalsBufferSize = accels.clusterVertexNormalsBuffer.GetBytes();
    stats.allocated.m_clasSize = accels.clasBuffer.GetBytes();
    stats.allocated.m_clusterDataSize = m_clustersBuffer.GetBytes() + accels.clusterShadingDataBuffer.GetBytes() + accels.clasPtrsBuffer.GetBytes();
    stats.allocated.m_blasSize = accels.blasBuffer.GetBytes();
    stats.allocated.m_blasScratchSize = m_createBlasSizeInfo.scratchSizeInBytes;

    m_buildAccelFrameIndex++;

    // Log final statistics
    RTXMG_LOG(str::format("RTX MegaGeo: BuildAccel COMPLETE - clusters=", counters.clusters,
        " desiredClusters=", counters.desiredClusters,
        " blasPtrsBuffer bytes=", accels.blasPtrsBuffer.GetBytes(),
        " blasBuffer bytes=", accels.blasBuffer.GetBytes()));

#if RTXMG_CHRONO_TIMING
    {
        auto chronoEnd = std::chrono::high_resolution_clock::now();
        float totalMs = std::chrono::duration_cast<std::chrono::microseconds>(chronoEnd - chronoStart).count() * 0.001f;
        Logger::info(str::format(">>> RTXMG CHRONO: BuildAccel TOTAL=", totalMs, "ms clusters=", counters.clusters));
    }
#endif
}

void ClusterAccelBuilder::ProcessDeferredDestruction(uint32_t currentFrame)
{
    // Remove entries that are old enough to be safely destroyed
    // The GPU should be done with them after kDeferredDestructionFrames
    size_t sizeBefore = m_deferredDestructionQueue.size();
    m_deferredDestructionQueue.erase(
        std::remove_if(m_deferredDestructionQueue.begin(), m_deferredDestructionQueue.end(),
            [currentFrame](const DeferredBuffers& deferred) {
                bool shouldDestroy = (currentFrame - deferred.frameQueued) >= kDeferredDestructionFrames;
                return shouldDestroy;
            }),
        m_deferredDestructionQueue.end());
}

