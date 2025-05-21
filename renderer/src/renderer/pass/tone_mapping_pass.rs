use std::ffi::CString;

use anyhow::Result;
use ash::vk;

use crate::renderer::{Renderer, render_images::RenderImages, vulkan_state::VulkanState};

/// A struct that represents the tone mapping pass.
pub struct ToneMappingPass {
    #[allow(dead_code)]
    descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}
impl ToneMappingPass {
    /// Creates a new instance of the ToneMappingPass struct.
    pub fn new(state: &VulkanState, render_images: &RenderImages) -> Result<Self> {
        // Create descriptor set layout
        let descriptor_set_layout = {
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let descriptor_set_layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe {
                state
                    .device
                    .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?
            }
        };
        // Create descriptor pool
        let descriptor_pool = {
            let descriptor_pool_size = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(Renderer::MAX_FRAMES_IN_FLIGHT as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(Renderer::MAX_FRAMES_IN_FLIGHT as u32),
            ];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&descriptor_pool_size)
                .max_sets(Renderer::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe {
                state
                    .device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)?
            }
        };
        // Create descriptor sets
        let descriptor_sets = {
            let set_layouts = [descriptor_set_layout]
                .into_iter()
                .cycle()
                .take(Renderer::MAX_FRAMES_IN_FLIGHT as usize)
                .collect::<Vec<_>>();
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe {
                state
                    .device
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)?
            }
        };
        // Update descriptor sets
        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            let input_image_info = [vk::DescriptorImageInfo::default()
                .image_view(render_images.linear_scene_image_views[i])
                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)];
            let output_image_info = [vk::DescriptorImageInfo::default()
                .image_view(render_images.after_tone_mapping_image_views[i])
                .image_layout(vk::ImageLayout::GENERAL)];
            let write_descriptor_sets = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_sets[i])
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_binding(0)
                    .image_info(&input_image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(*descriptor_set)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_binding(1)
                    .image_info(&output_image_info),
            ];
            unsafe {
                state
                    .device
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }
        // Create compute pipeline
        let (pipeline_layout, pipeline) = {
            // Create shader stage create infos
            let compute_shader_module = {
                let code =
                    include_bytes!(concat!(env!("OUT_DIR"), "/shaders/tone_mapping.comp.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { state.device.create_shader_module(&create_info, None)? }
            };
            let main_function_name = CString::new("main")?;
            let shader_stages = [vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(compute_shader_module)
                .name(&main_function_name)];

            // Create pipeline layout
            let set_layouts = [descriptor_set_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let pipeline_layout = unsafe {
                state
                    .device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)?
            };

            // Create compute pipeline create info
            let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(shader_stages[0])
                .layout(pipeline_layout);
            let compute_pipeline = unsafe {
                state
                    .device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[compute_pipeline_create_info],
                        None,
                    )
                    .expect("Failed to create compute pipeline")
            }[0];

            // Destroy shader modules
            unsafe {
                state
                    .device
                    .destroy_shader_module(compute_shader_module, None);
            }

            (pipeline_layout, compute_pipeline)
        };

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
        })
    }

    /// Update render images.
    pub fn update_render_images(&mut self, state: &VulkanState, render_images: &RenderImages) {
        // Update descriptor sets
        for i in 0..Renderer::MAX_FRAMES_IN_FLIGHT {
            let input_image_info = [vk::DescriptorImageInfo::default()
                .image_view(render_images.linear_scene_image_views[i])
                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)];
            let output_image_info = [vk::DescriptorImageInfo::default()
                .image_view(render_images.after_tone_mapping_image_views[i])
                .image_layout(vk::ImageLayout::GENERAL)];
            let write_descriptor_sets = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_binding(0)
                    .image_info(&input_image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_binding(1)
                    .image_info(&output_image_info),
            ];
            unsafe {
                state
                    .device
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }
    }

    /// Record the command buffer for the tone mapping pass.
    pub fn cmd_draw(
        &self,
        state: &VulkanState,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
    ) {
        // Memory barrier
        // - linear_scene_images[image_index] ColorAttachmentOptimal -> ReadOnlyOptimal
        // - after_tone_mapping_image[image_index] ShaderReadOnlyOptimal -> General
        let image_memory_barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .image(render_images.linear_scene_images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(render_images.after_tone_mapping_images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        unsafe {
            state.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&image_memory_barriers),
            );
        }

        // bind compute pipeline
        unsafe {
            state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        }

        // bind descriptor sets
        unsafe {
            state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
        }

        // Dispatch compute shader
        let x = state.swapchain.extent.width.div_ceil(8);
        let y = state.swapchain.extent.height.div_ceil(8);
        unsafe {
            state.device.cmd_dispatch(command_buffer, x, y, 1);
        }
    }

    /// Destroy the tone mapping pass.
    pub fn destroy(&mut self, state: &VulkanState) {
        unsafe {
            state
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            state.device.destroy_pipeline(self.pipeline, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
