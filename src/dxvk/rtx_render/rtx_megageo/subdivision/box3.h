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
 * THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "../../../util/util_vector.h"
#include "../../../util/util_math.h"
#include <limits>
#include <algorithm>

// box3 type definition extracted to avoid circular dependencies between
// shape.h and subdivision_surface.h
namespace donut {
  namespace math {
    struct box3 {
      float m_mins[3];
      float m_maxs[3];

      box3() {
        m_mins[0] = m_mins[1] = m_mins[2] = std::numeric_limits<float>::max();
        m_maxs[0] = m_maxs[1] = m_maxs[2] = std::numeric_limits<float>::lowest();
      }

      box3(float minx, float miny, float minz, float maxx, float maxy, float maxz) {
        m_mins[0] = minx; m_mins[1] = miny; m_mins[2] = minz;
        m_maxs[0] = maxx; m_maxs[1] = maxy; m_maxs[2] = maxz;
      }

      bool empty() const {
        return m_mins[0] >= m_maxs[0] || m_mins[1] >= m_maxs[1] || m_mins[2] >= m_maxs[2];
      }

      static box3 makeEmpty() {
        return box3();
      }

      box3& operator|=(const dxvk::Vector3& point) {
        m_mins[0] = std::min(m_mins[0], point.x);
        m_mins[1] = std::min(m_mins[1], point.y);
        m_mins[2] = std::min(m_mins[2], point.z);
        m_maxs[0] = std::max(m_maxs[0], point.x);
        m_maxs[1] = std::max(m_maxs[1], point.y);
        m_maxs[2] = std::max(m_maxs[2], point.z);
        return *this;
      }

      // Transform box by matrix - transforms all 8 corners and rebuilds AABB
      box3 operator*(const dxvk::Matrix4& matrix) const {
        box3 result = makeEmpty();

        // Transform all 8 corners of the box
        for (int i = 0; i < 8; ++i) {
          float x = (i & 1) ? m_maxs[0] : m_mins[0];
          float y = (i & 2) ? m_maxs[1] : m_mins[1];
          float z = (i & 4) ? m_maxs[2] : m_mins[2];

          // Transform point by matrix
          float tx = matrix.data[0][0] * x + matrix.data[0][1] * y + matrix.data[0][2] * z + matrix.data[0][3];
          float ty = matrix.data[1][0] * x + matrix.data[1][1] * y + matrix.data[1][2] * z + matrix.data[1][3];
          float tz = matrix.data[2][0] * x + matrix.data[2][1] * y + matrix.data[2][2] * z + matrix.data[2][3];

          result |= dxvk::Vector3(tx, ty, tz);
        }

        return result;
      }
    };
  }
}

using box3 = donut::math::box3;
