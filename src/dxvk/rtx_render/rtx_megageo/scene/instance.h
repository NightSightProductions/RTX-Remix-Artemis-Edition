/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include "../../../util/util_math.h"
#include <string>
#include <memory>
#include <vector>

// Geometry info structure - represents a single geometry in a mesh
struct GeometryInfo {
    uint32_t globalGeometryIndex = 0;  // Global index for this geometry
    std::string name;
};

// MeshInfo structure for mesh metadata
struct MeshInfo {
    std::string name;
    uint32_t geometryCount = 0;
    std::vector<std::shared_ptr<GeometryInfo>> geometries;

    MeshInfo* get() { return this; }
    const MeshInfo* get() const { return this; }
};

// MeshInstance wrapper - wraps MeshInfo pointer for scene graph
class MeshInstance {
public:
    MeshInstance(std::shared_ptr<MeshInfo> info) : m_info(info) {}

    MeshInfo* get() const { return m_info.get(); }
    MeshInfo& GetMesh() { return *m_info; }
    const MeshInfo& GetMesh() const { return *m_info; }
    MeshInfo* operator->() const { return m_info.get(); }

private:
    std::shared_ptr<MeshInfo> m_info;
};

// Instance structure for mesh instances in the scene
struct Instance {
    uint32_t meshID = 0;                     // ID of the mesh this instance references
    dxvk::Matrix4 localToWorld;              // Transform from local to world space (4x4 matrix)
    std::shared_ptr<MeshInstance> meshInstance; // Pointer to mesh instance wrapper

    Instance() : localToWorld(dxvk::Matrix4()) {}
};
