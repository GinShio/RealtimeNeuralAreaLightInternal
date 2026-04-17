use std::ffi::CString;

use anyhow::Result;
use ash::vk;

use crate::renderer::{Renderer, render_images::RenderImages, vulkan_state::VulkanState};
use ash::vk::TaggedStructure;

/// A struct that represents the copy to swapchain pass.
pub struct CopyToSwapchainPass {
    sampler: vk::Sampler,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}
impl CopyToSwapchainPass {
    /// Creates a new instance of the CopyToSwapchainPass struct.
    pub fn new(state: &VulkanState, render_images: &RenderImages) -> Result<Self> {
        // Create sampler
        let sampler = {
            let sampler_create_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK);
            unsafe { state.device.create_sampler(&sampler_create_info, None)? }
        };
        // Create descriptor set layout
        let descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
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
            let descriptor_pool_size = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(Renderer::IMAGE_COUNT as u32)];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&descriptor_pool_size)
                .max_sets(Renderer::IMAGE_COUNT as u32);
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
                .take(Renderer::IMAGE_COUNT)
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
                .image_view(render_images.after_tone_mapping_image_views[i])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(sampler)];
            let write_descriptor_sets = [vk::WriteDescriptorSet::default()
                .dst_set(*descriptor_set)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(0)
                .image_info(&input_image_info)];
            unsafe {
                state
                    .device
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }
        // Create pipeline
        let (pipeline_layout, pipeline) = {
            // Create shader stage create infos
            let vertex_shader_module = {
                let code = include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/copy_to_swapchain.vert.spv"
                ));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { state.device.create_shader_module(&create_info, None)? }
            };
            let fragment_shader_module = {
                let code = include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/copy_to_swapchain.frag.spv"
                ));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { state.device.create_shader_module(&create_info, None)? }
            };
            let main_function_name = CString::new("main")?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name),
            ];

            // Create vertex input state create info
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

            // Create input assembly state info
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            // Dynamic state create info
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            // Create viewport state create info
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            // Create rasterization state create info
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);

            // Create multisample state create info
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // Create color blend attachment states
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)];

            // Create color blend state create info
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(&color_blend_attachment_states);

            // Create pipeline rendering create info
            let rendering_formats = [state.swapchain.format];
            let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&rendering_formats);

            // Create pipeline layout
            let set_layouts = [descriptor_set_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let pipeline_layout = unsafe {
                state
                    .device
                    .create_pipeline_layout(&pipeline_layout_create_info, None)?
            };

            // Create graphics pipeline create info
            let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .push(&mut pipeline_rendering)
                .layout(pipeline_layout);
            let pipeline = unsafe {
                state
                    .device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[graphics_pipeline_create_info],
                        None,
                    )
                    .expect("Failed to create graphics pipeline")
            }[0];

            // Destroy shader modules
            unsafe {
                state
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
                state
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
            }

            (pipeline_layout, pipeline)
        };

        Ok(Self {
            sampler,
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
        for i in 0..Renderer::IMAGE_COUNT {
            let input_image_info = [vk::DescriptorImageInfo::default()
                .image_view(render_images.after_tone_mapping_image_views[i])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(self.sampler)];
            let write_descriptor_sets = [vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(0)
                .image_info(&input_image_info)];
            unsafe {
                state
                    .device
                    .update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }
    }

    /// Record the command buffer for the copy to swapchain pass.
    pub fn cmd_draw(
        &self,
        state: &VulkanState,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
    ) {
        // Memory barrier
        // - after_tone_mapping_image[image_index] ColorAttachmentOptimal -> ShaderReadOnlyOptimal
        // - state.swapchain.images[image_index] PresentSrcKHR -> ColorAttachmentOptimal
        let image_memory_barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(render_images.after_tone_mapping_images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2KHR::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(state.swapchain.images[image_index])
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

        // Begin rendering
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(state.swapchain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: state.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            state
                .device
                .cmd_begin_rendering(command_buffer, &rendering_info);
        }

        // Bind pipeline
        unsafe {
            state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        }

        // Set viewport and scissor
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(state.swapchain.extent.width as f32)
            .height(state.swapchain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(state.swapchain.extent);
        unsafe {
            state
                .device
                .cmd_set_viewport(command_buffer, 0, &[viewport]);
            state.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        // bind descriptor set
        unsafe {
            state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
        }

        // Draw
        unsafe {
            state.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        // End rendering
        unsafe {
            state.device.cmd_end_rendering(command_buffer);
        }

        // Memory barrier
        // - state.swapchain.images[image_index] ColorAttachmentOptimal -> PresentSrcKHR
        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2KHR::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2KHR::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(state.swapchain.images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            state.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&[image_memory_barrier]),
            );
        }
    }

    /// Destroy the copy to swapchain pass.
    pub fn destroy(&mut self, state: &VulkanState) {
        unsafe {
            state.device.destroy_sampler(self.sampler, None);
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
