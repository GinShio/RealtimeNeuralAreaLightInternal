use std::ffi::CString;

use anyhow::Result;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::renderer::{render_images::RenderImages, vulkan_state::VulkanState};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}
impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            // position
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: std::mem::offset_of!(Self, pos) as u32,
            },
            // color
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(Self, color) as u32,
            },
        ]
    }
}

/// A struct that represents the scene pass.
pub struct ScenePass {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,
}
impl ScenePass {
    /// Creates a new instance of the ScenePass struct.
    pub fn new(state: &mut VulkanState) -> Result<Self> {
        // Create graphics pipeline
        let (pipeline_layout, pipeline) = {
            // Create shader stage create infos
            let vertex_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/scene.vert.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { state.device.create_shader_module(&create_info, None)? }
            };
            let fragment_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/scene.frag.spv"));
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
            let binding_description = Vertex::get_binding_descriptions();
            let attribute_descriptions = Vertex::get_attribute_descriptions();
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_description)
                .vertex_attribute_descriptions(&attribute_descriptions);

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
                .cull_mode(vk::CullModeFlags::BACK)
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
            let rendering_formats = [vk::Format::R8G8B8A8_UNORM];
            let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&rendering_formats);

            // Create pipeline layout
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&[]);
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
                .push_next(&mut pipeline_rendering)
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

        // create vertex buffer
        let vertices = vec![
            Vertex {
                pos: [0.0, 0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [-0.5, -0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5],
                color: [0.0, 0.0, 1.0],
            },
        ];
        let (vertex_buffer, vertex_buffer_allocation) = {
            let buffer_size = (std::mem::size_of::<Vertex>() * vertices.len()) as u64;

            // create staging buffer
            let staging_buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging_buffer = unsafe {
                state
                    .device
                    .create_buffer(&staging_buffer_create_info, None)?
            };

            // Allocate memory for the staging buffer
            let staging_buffer_requirements =
                unsafe { state.device.get_buffer_memory_requirements(staging_buffer) };
            let mut staging_buffer_allocation =
                state.allocator().allocate(&AllocationCreateDesc {
                    name: "vertex staging buffer",
                    requirements: staging_buffer_requirements,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })?;

            // Bind the staging buffer memory
            unsafe {
                state.device.bind_buffer_memory(
                    staging_buffer,
                    staging_buffer_allocation.memory(),
                    staging_buffer_allocation.offset(),
                )?;
            }

            // Map the staging buffer memory
            let data = staging_buffer_allocation
                .mapped_slice_mut()
                .ok_or_else(|| {
                    panic!("Failed to map staging buffer memory");
                })?;
            data.copy_from_slice(bytemuck::cast_slice(&vertices));

            // Create vertex buffer
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let vertex_buffer = unsafe { state.device.create_buffer(&buffer_create_info, None)? };

            // Allocate memory for the vertex buffer
            let vertex_buffer_requirements =
                unsafe { state.device.get_buffer_memory_requirements(vertex_buffer) };
            let vertex_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
                name: "vertex buffer",
                requirements: vertex_buffer_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            // Bind the vertex buffer memory
            unsafe {
                state.device.bind_buffer_memory(
                    vertex_buffer,
                    vertex_buffer_allocation.memory(),
                    vertex_buffer_allocation.offset(),
                )?;
            }

            // Create a command buffer
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(state.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = unsafe {
                state
                    .device
                    .allocate_command_buffers(&command_buffer_allocate_info)?
            }[0];

            // Record copy command to the command buffer
            unsafe {
                state
                    .device
                    .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
                state.device.cmd_copy_buffer2(
                    command_buffer,
                    &vk::CopyBufferInfo2::default()
                        .src_buffer(staging_buffer)
                        .dst_buffer(vertex_buffer)
                        .regions(&[vk::BufferCopy2::default()
                            .src_offset(0)
                            .dst_offset(0)
                            .size(buffer_size)]),
                );
                state.device.end_command_buffer(command_buffer)?;
            }

            // Create a fence
            let fence_create_info = vk::FenceCreateInfo::default();
            let fence = unsafe { state.device.create_fence(&fence_create_info, None)? };

            // Submit the command buffer
            let buffers_for_submission = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
            unsafe {
                state
                    .device
                    .queue_submit(state.queue, &[submit_info], fence)?;
                state.device.wait_for_fences(&[fence], true, u64::MAX)?;
            }

            // Destroy the fence and command buffer
            unsafe {
                state.device.destroy_fence(fence, None);
                state
                    .device
                    .free_command_buffers(state.command_pool, &[command_buffer]);
            }

            // Destroy the staging buffer
            state.allocator().free(staging_buffer_allocation)?;
            unsafe {
                state.device.destroy_buffer(staging_buffer, None);
            }

            // Return the vertex buffer and its memory
            (vertex_buffer, vertex_buffer_allocation)
        };

        Ok(Self {
            pipeline_layout,
            pipeline,
            vertices,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),
        })
    }

    /// Record the command buffer for the scene pass.
    pub fn cmd_draw(
        &self,
        state: &VulkanState,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
    ) {
        // Memory barrier
        // - linear_scene_images[image_index] ReadOnlyOptimal -> ColorAttachmentOptimal
        let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2KHR::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(render_images.linear_scene_images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        unsafe {
            state.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&image_memory_barriers),
            );
        }

        // Begin rendering
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(render_images.linear_scene_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.2, 0.3, 1.0],
                },
            })];
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
            .y(state.swapchain.extent.height as f32)
            .width(state.swapchain.extent.width as f32)
            .height(-(state.swapchain.extent.height as f32))
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

        // Bind vertex buffer
        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0];
        unsafe {
            state
                .device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
        }

        // Draw
        unsafe {
            state
                .device
                .cmd_draw(command_buffer, self.vertices.len() as u32, 1, 0, 0);
        }

        // End rendering
        unsafe {
            state.device.cmd_end_rendering(command_buffer);
        }
    }

    /// Destroy the scene pass.
    pub fn destroy(&mut self, state: &mut VulkanState) -> Result<()> {
        unsafe {
            state.device.destroy_pipeline(self.pipeline, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            state.device.destroy_buffer(self.vertex_buffer, None);
            state
                .allocator()
                .free(self.vertex_buffer_allocation.take().unwrap())?;
        }
        if let Some(allocation) = self.vertex_buffer_allocation.take() {
            state.allocator().free(allocation)?;
        }

        Ok(())
    }
}
