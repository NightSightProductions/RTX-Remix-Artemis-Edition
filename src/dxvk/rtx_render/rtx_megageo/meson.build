# RTX Mega Geometry (RTX MG) Integration
# Based on NVIDIA's RTX MG sample, adapted for RTX Remix

# =============================================================================
# OSD Lite (OpenSubdiv) - Vendored Library (mirroring the sample)
# =============================================================================

osd_lite_sources = files([
  # BFR (Base Face Representation) - 16 files
  'osd_lite/opensubdiv/bfr/faceSurface.cpp',
  'osd_lite/opensubdiv/bfr/faceTopology.cpp',
  'osd_lite/opensubdiv/bfr/faceVertex.cpp',
  'osd_lite/opensubdiv/bfr/hash.cpp',
  'osd_lite/opensubdiv/bfr/irregularPatchBuilder.cpp',
  'osd_lite/opensubdiv/bfr/parameterization.cpp',
  'osd_lite/opensubdiv/bfr/patchTree.cpp',
  'osd_lite/opensubdiv/bfr/patchTreeBuilder.cpp',
  'osd_lite/opensubdiv/bfr/refinerSurfaceFactory.cpp',
  'osd_lite/opensubdiv/bfr/regularPatchBuilder.cpp',
  'osd_lite/opensubdiv/bfr/surface.cpp',
  'osd_lite/opensubdiv/bfr/surfaceData.cpp',
  'osd_lite/opensubdiv/bfr/surfaceFactory.cpp',
  'osd_lite/opensubdiv/bfr/surfaceFactoryCache.cpp',
  'osd_lite/opensubdiv/bfr/tessellation.cpp',
  'osd_lite/opensubdiv/bfr/vertexDescriptor.cpp',

  # FAR (Feature Adaptive Representation) - 17 files
  'osd_lite/opensubdiv/far/bilinearPatchBuilder.cpp',
  'osd_lite/opensubdiv/far/catmarkPatchBuilder.cpp',
  'osd_lite/opensubdiv/far/error.cpp',
  'osd_lite/opensubdiv/far/loopPatchBuilder.cpp',
  'osd_lite/opensubdiv/far/patchBasis.cpp',
  'osd_lite/opensubdiv/far/patchBuilder.cpp',
  'osd_lite/opensubdiv/far/patchDescriptor.cpp',
  'osd_lite/opensubdiv/far/patchMap.cpp',
  'osd_lite/opensubdiv/far/patchTable.cpp',
  'osd_lite/opensubdiv/far/patchTableFactory.cpp',
  'osd_lite/opensubdiv/far/ptexIndices.cpp',
  'osd_lite/opensubdiv/far/stencilBuilder.cpp',
  'osd_lite/opensubdiv/far/stencilTable.cpp',
  'osd_lite/opensubdiv/far/stencilTableFactory.cpp',
  'osd_lite/opensubdiv/far/topologyDescriptor.cpp',
  'osd_lite/opensubdiv/far/topologyRefiner.cpp',
  'osd_lite/opensubdiv/far/topologyRefinerFactory.cpp',

  # SDC (Subdivision) - 3 files
  'osd_lite/opensubdiv/sdc/crease.cpp',
  'osd_lite/opensubdiv/sdc/typeTraits.cpp',

  # TMR (Tessellation Mesh Refinement) - 8 files
  'osd_lite/opensubdiv/tmr/neighborhood.cpp',
  'osd_lite/opensubdiv/tmr/neighborhoodBuilder.cpp',
  'osd_lite/opensubdiv/tmr/subdivisionPlan.cpp',
  'osd_lite/opensubdiv/tmr/subdivisionPlanBuilder.cpp',
  'osd_lite/opensubdiv/tmr/surfaceTable.cpp',
  'osd_lite/opensubdiv/tmr/surfaceTableFactory.cpp',
  'osd_lite/opensubdiv/tmr/topologyMap.cpp',
  'osd_lite/opensubdiv/tmr/unorderedSubset.cpp',

  # VTR (Vertex Topology Representation) - 7 files
  'osd_lite/opensubdiv/vtr/fvarLevel.cpp',
  'osd_lite/opensubdiv/vtr/fvarRefinement.cpp',
  'osd_lite/opensubdiv/vtr/level.cpp',
  'osd_lite/opensubdiv/vtr/quadRefinement.cpp',
  'osd_lite/opensubdiv/vtr/refinement.cpp',
  'osd_lite/opensubdiv/vtr/sparseSelector.cpp',
  'osd_lite/opensubdiv/vtr/triRefinement.cpp',

  # Version - 1 file
  'osd_lite/opensubdiv/version.cpp',
])

osd_lite_include = include_directories('osd_lite')

# Build OSD Lite as static library (matching the sample)
osd_lite_lib = static_library('osd_lite', osd_lite_sources,
  include_directories : osd_lite_include,
  cpp_args : [
    '-DNOMINMAX',
    '-D_USE_MATH_DEFINES'
  ],
  override_options : ['cpp_std=c++20']
)

osd_lite_dep = declare_dependency(
  link_with : osd_lite_lib,
  include_directories : osd_lite_include
)

# =============================================================================
# RTX MG Core Modules
# =============================================================================

rtx_mg_include = include_directories('.')

rtx_mg_sources = files([
  # NVRHI Adapter (4 files - critical for DXVK integration)
  'nvrhi_adapter/nvrhi_dxvk_device.cpp',
  'nvrhi_adapter/nvrhi_dxvk_command_list.cpp',
  'nvrhi_adapter/nvrhi_scratch_manager.cpp',
  'nvrhi_adapter/donut_adapter.cpp',

  # High-Level Wrapper (1 file - RTX Remix integration point)
  'rtx_megageo_builder.cpp',

  # Cluster Builder (1 file)
  'cluster_builder/cluster_accel_builder.cpp',

  # HIZ (Hierarchical Z-Buffer) (2 files)
  'hiz/hiz_buffer.cpp',
  'hiz/zbuffer.cpp',

  # Subdivision (4 files)
  'subdivision/shape.cpp',
  'subdivision/subdivision_surface.cpp',
  'subdivision/topology_cache.cpp',
  'subdivision/topology_map.cpp',

  # Scene (2 files - RTX MG scene integration)
  'scene/camera.cpp',
  'scene/rtxmg_scene.cpp',

  # Utils (1 file - buffer helpers)
  'utils/buffer.cpp',

  # Profiler (1 file - stub implementation)
  'profiler/profiler_stub.cpp',
])

# Build RTX MG as a static library with access to OSD Lite includes
# Note: Needs all DXVK dependencies including USD for RTX Remix headers
# Note: include_directories('../..') adds src/dxvk/ so dxvk_include.h can be found
# Note: Using C++20 to match osd_lite (which uses 'using enum' and other C++20 features)
rtx_mg_lib = static_library('rtx_mg', rtx_mg_sources,
  include_directories : [rtx_mg_include, include_directories('../..'), dxvk_include_path, remix_api_include_path, dxvk_shader_include_path],
  dependencies : [osd_lite_dep, usd_dep],
  override_options : ['cpp_std=c++20']
)

# RTX MG Shaders - Cluster Builder
# Note: All HLSL files removed - actual shaders come from pre-compiled Slang versions via donut_adapter.cpp
# The shaders are: copy_cluster_offset, compute_cluster_tiling, fill_clusters, fill_blas_from_clas_args, patch_cluster_blas_addresses
rtx_mg_cluster_shaders = files([])

# RTX MG Shaders - HIZ
rtx_mg_hiz_shaders = files([
  'shaders/hiz/hiz_display.hlsl',
  'shaders/hiz/hiz_pass1.hlsl',
  'shaders/hiz/hiz_pass2.hlsl',
  'shaders/hiz/zbuffer_display.hlsl',
  'shaders/hiz/zbuffer_minmax.hlsl',
])

# RTX MG Dependency (to be used by main dxvk build)
# Note: osd_lite include is NOT exposed to avoid shadowing DXVK's version.h
# RTX MG and OSD Lite are provided as linked libraries
rtx_mg_dep = declare_dependency(
  include_directories : rtx_mg_include,
  link_with : [rtx_mg_lib, osd_lite_lib]
)

# Export sources and shaders to parent build
# These will be added to dxvk_src and shader lists in parent meson.build
