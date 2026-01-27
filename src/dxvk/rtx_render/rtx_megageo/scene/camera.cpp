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

#include "camera.h"
#include <cmath>

namespace dxvk {

Vector3 Camera::GetDirection() const {
    Vector3 dir = m_lookat - m_eye;
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (len > 0.0f) {
        return Vector3(dir.x / len, dir.y / len, dir.z / len);
    }
    return Vector3(0, 0, -1);
}

void Camera::Translate(Vector3 const& v) {
    m_eye.x += v.x; m_eye.y += v.y; m_eye.z += v.z;
    m_lookat.x += v.x; m_lookat.y += v.y; m_lookat.z += v.z;
    m_changed = true;
}

void Camera::Rotate(float yaw, float pitch, float roll) {
    // Implementation for camera rotation
    m_changed = true;
}

void Camera::Roll(float speed) {
    m_changed = true;
}

void Camera::Dolly(float factor) {
    Vector3 dir = Vector3(m_lookat.x - m_eye.x, m_lookat.y - m_eye.y, m_lookat.z - m_eye.z);
    m_eye.x += dir.x * factor;
    m_eye.y += dir.y * factor;
    m_eye.z += dir.z * factor;
    m_changed = true;
}

void Camera::Pan(Vector2 speed) {
    m_changed = true;
}

void Camera::Zoom(const float factor) {
    m_fovY *= factor;
    m_changed = true;
}

void Camera::ComputeBasis(Vector3& u, Vector3& v, Vector3& w) const {
    // w = normalize(eye - lookat)
    w = Vector3(m_eye.x - m_lookat.x, m_eye.y - m_lookat.y, m_eye.z - m_lookat.z);
    float wLen = std::sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    if (wLen > 0.0f) {
        w.x /= wLen; w.y /= wLen; w.z /= wLen;
    }

    // u = normalize(cross(up, w))
    u = Vector3(
        m_up.y * w.z - m_up.z * w.y,
        m_up.z * w.x - m_up.x * w.z,
        m_up.x * w.y - m_up.y * w.x
    );
    float uLen = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    if (uLen > 0.0f) {
        u.x /= uLen; u.y /= uLen; u.z /= uLen;
    }

    // v = cross(w, u)
    v = Vector3(
        w.y * u.z - w.z * u.y,
        w.z * u.x - w.x * u.z,
        w.x * u.y - w.y * u.x
    );
}

Matrix4 Camera::GetViewMatrix() const {
    Vector3 u, v, w;
    ComputeBasis(u, v, w);

    // Create view matrix (lookAt)
    Matrix4 mat;
    mat.data[0][0] = u.x; mat.data[0][1] = v.x; mat.data[0][2] = w.x; mat.data[0][3] = 0.0f;
    mat.data[1][0] = u.y; mat.data[1][1] = v.y; mat.data[1][2] = w.y; mat.data[1][3] = 0.0f;
    mat.data[2][0] = u.z; mat.data[2][1] = v.z; mat.data[2][2] = w.z; mat.data[2][3] = 0.0f;
    mat.data[3][0] = -(u.x * m_eye.x + u.y * m_eye.y + u.z * m_eye.z);
    mat.data[3][1] = -(v.x * m_eye.x + v.y * m_eye.y + v.z * m_eye.z);
    mat.data[3][2] = -(w.x * m_eye.x + w.y * m_eye.y + w.z * m_eye.z);
    mat.data[3][3] = 1.0f;

    return mat;
}

Matrix4 Camera::GetProjectionMatrix() const {
    float tanHalfFovy = std::tan(m_fovY * 0.5f * 3.14159265f / 180.0f);

    Matrix4 mat;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            mat.data[i][j] = 0.0f;

    mat.data[0][0] = 1.0f / (m_aspectRatio * tanHalfFovy);
    mat.data[1][1] = 1.0f / tanHalfFovy;
    mat.data[2][2] = m_zFar / (m_zFar - m_zNear);
    mat.data[2][3] = 1.0f;
    mat.data[3][2] = -(m_zFar * m_zNear) / (m_zFar - m_zNear);

    return mat;
}

Matrix4 Camera::GetViewProjectionMatrix() const {
    // Simple matrix multiplication
    Matrix4 proj = GetProjectionMatrix();
    Matrix4 view = GetViewMatrix();
    Matrix4 result;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.data[i][j] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                result.data[i][j] += proj.data[i][k] * view.data[k][j];
            }
        }
    }

    return result;
}

void Camera::SetEye(Vector3 eye) {
    m_eye = eye;
    m_changed = true;
}

void Camera::SetLookat(Vector3 lookat) {
    m_lookat = lookat;
    m_changed = true;
}

void Camera::SetUp(Vector3 up) {
    m_up = up;
    m_changed = true;
}

void Camera::SetFovY(float fovy) {
    m_fovY = fovy;
    m_changed = true;
}

void Camera::SetAspectRatio(float ar) {
    m_aspectRatio = ar;
    m_changed = true;
}

void Camera::SetNear(float nearPlane) {
    m_zNear = nearPlane;
    m_changed = true;
}

void Camera::SetFar(float farPlane) {
    m_zFar = farPlane;
    m_changed = true;
}

} // namespace dxvk
