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

#include "nvrhi_types.h"
#include "../../dxvk_compute.h"
#include "../../dxvk_shader.h"

namespace dxvk {

  // NVRHI IBindingLayout implementation storing binding descriptions
  // When registerSpaceIsDescriptorSet is true, also holds a VkDescriptorSetLayout
  class NvrhiDxvkBindingLayout : public nvrhi::IBindingLayout {
  public:
    NvrhiDxvkBindingLayout(const nvrhi::BindingLayoutDesc& desc)
      : m_desc(desc)
      , m_vkDescriptorSetLayout(VK_NULL_HANDLE)
      , m_vkDevice(VK_NULL_HANDLE)
    {
    }

    ~NvrhiDxvkBindingLayout() {
      // Destroy the VkDescriptorSetLayout if we created one
      if (m_vkDescriptorSetLayout != VK_NULL_HANDLE && m_vkDevice != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_vkDevice, m_vkDescriptorSetLayout, nullptr);
        m_vkDescriptorSetLayout = VK_NULL_HANDLE;
      }
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      // VkDescriptorSetLayout doesn't have a dedicated ObjectType, use getVkDescriptorSetLayout() instead
      return nvrhi::Object();
    }

    const nvrhi::BindingLayoutDesc& getDesc() const { return m_desc; }

    // Set the VkDescriptorSetLayout (called by NvrhiDxvkDevice when registerSpaceIsDescriptorSet is true)
    void setVkDescriptorSetLayout(VkDescriptorSetLayout layout, VkDevice device) {
      m_vkDescriptorSetLayout = layout;
      m_vkDevice = device;
    }

    VkDescriptorSetLayout getVkDescriptorSetLayout() const { return m_vkDescriptorSetLayout; }
    bool hasVkDescriptorSetLayout() const { return m_vkDescriptorSetLayout != VK_NULL_HANDLE; }

  private:
    nvrhi::BindingLayoutDesc m_desc;
    VkDescriptorSetLayout m_vkDescriptorSetLayout;
    VkDevice m_vkDevice;  // Needed for cleanup
  };

  // NVRHI IBindingSet implementation storing resource bindings
  class NvrhiDxvkBindingSet : public nvrhi::IBindingSet {
  public:
    NvrhiDxvkBindingSet(const nvrhi::BindingSetDesc& desc, nvrhi::IBindingLayout* layout)
      : m_desc(desc)
      , m_layout(layout)
    {
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      return nvrhi::Object();
    }

    const nvrhi::BindingSetDesc& getDesc() const { return m_desc; }
    nvrhi::IBindingLayout* getLayout() const { return m_layout; }

  private:
    nvrhi::BindingSetDesc m_desc;
    nvrhi::IBindingLayout* m_layout;
  };

  // NVRHI IComputePipeline implementation wrapping DXVK compute pipeline
  class NvrhiDxvkComputePipeline : public nvrhi::IComputePipeline {
  public:
    NvrhiDxvkComputePipeline(
      const Rc<DxvkShader>& shader,
      const nvrhi::ComputePipelineDesc& desc)
      : m_shader(shader)
      , m_desc(desc)
    {
    }

    nvrhi::Object getNativeObject(nvrhi::ObjectType type) override {
      return nvrhi::Object();
    }

    const nvrhi::ComputePipelineDesc& getDesc() const { return m_desc; }
    const Rc<DxvkShader>& getShader() const { return m_shader; }

  private:
    Rc<DxvkShader> m_shader;
    nvrhi::ComputePipelineDesc m_desc;
  };

} // namespace dxvk
