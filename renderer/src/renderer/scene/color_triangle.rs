use anyhow::Result;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::renderer::{
    network::Network, render_images::RenderImages, scene::Scene, texture_manager::TextureManager,
    utils, vertex::Vertex, vulkan_state::VulkanState,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    fill_color: [f32; 3],
    _padding0: u32,
    weight_offsets: [u32; 3],
    _padding1: u32,
    bias_offsets: [u32; 3],
    _padding2: u32,
}

pub struct ColorTriangleScene {
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,

    network_buffer: vk::Buffer,
    network_buffer_allocation: Option<Allocation>,
    network_weight_offsets: Vec<u32>,
    network_bias_offsets: Vec<u32>,

    fill_color: [f32; 3],
}
impl ColorTriangleScene {
    pub fn new(state: &mut VulkanState, texture_manager: &mut TextureManager) -> Result<Box<Self>> {
        // Create descriptor set layout
        let descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(1);
            unsafe {
                state
                    .device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)?
            }
        };

        // Create descriptor set
        let descriptor_set = {
            let set_layouts = [descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_sets = unsafe { state.device.allocate_descriptor_sets(&allocate_info)? };
            descriptor_sets[0]
        };

        // Create graphics pipeline
        let (pipeline_layout, pipeline) = {
            utils::create_graphics_pipeline(
                state,
                texture_manager,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/color_triangle.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/color_triangle.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: std::mem::size_of::<PushConstants>() as u32,
                }],
                &[descriptor_set_layout],
            )?
        };

        // create vertex buffer
        let vertices = vec![
            Vertex {
                pos: [0.0, 0.5, 0.0],
                normal: [1.0, 0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                pos: [-0.5, -0.5, 0.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
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

        let network = Network::from_json(&state.cooperative_vector_fn, "./network/color.json")?;
        let (network_buffer, network_buffer_allocation) = {
            let buffer_size = network.data.len() as u64;

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
                    name: "network staging buffer",
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
            data[0..buffer_size as usize].copy_from_slice(bytemuck::cast_slice(&network.data));

            // Create network buffer
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let network_buffer = unsafe { state.device.create_buffer(&buffer_create_info, None)? };

            // Allocate memory for the network buffer
            let network_buffer_requirements =
                unsafe { state.device.get_buffer_memory_requirements(network_buffer) };
            let network_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
                name: "network buffer",
                requirements: network_buffer_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            // Bind the network buffer memory
            unsafe {
                state.device.bind_buffer_memory(
                    network_buffer,
                    network_buffer_allocation.memory(),
                    network_buffer_allocation.offset(),
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
                        .dst_buffer(network_buffer)
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
            (network_buffer, network_buffer_allocation)
        };

        // Update descriptor set
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info);
        unsafe {
            state
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[])
        };

        Ok(Box::new(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,

            pipeline_layout,
            pipeline,
            vertices,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),

            network_buffer,
            network_buffer_allocation: Some(network_buffer_allocation),
            network_weight_offsets: network.weight_offsets,
            network_bias_offsets: network.bias_offsets,

            fill_color: [1.0; 3],
        }))
    }
}
impl Scene for ColorTriangleScene {
    fn scene_name(&self) -> &'static str {
        "Color Triangle Scene"
    }

    fn ui(&mut self, ui: &imgui::Ui) {
        ui.color_edit3("fill color", &mut self.fill_color);
    }

    fn cmd_draw(
        &mut self,
        state: &VulkanState,
        texture_manager: &TextureManager,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
    ) {
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
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(render_images.depth_scene_image_views[image_index])
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: state.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
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

        // Bind descriptor sets
        let mut descriptor_sets = vec![];
        descriptor_sets.extend(texture_manager.descriptor_sets());
        descriptor_sets.push(self.descriptor_set);
        unsafe {
            state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &descriptor_sets,
                &[],
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

        // Push constants
        let push_constants = PushConstants {
            fill_color: self.fill_color,
            _padding0: 0,
            weight_offsets: [
                self.network_weight_offsets[0],
                self.network_weight_offsets[1],
                self.network_weight_offsets[2],
            ],
            _padding1: 0,
            bias_offsets: [
                self.network_bias_offsets[0],
                self.network_bias_offsets[1],
                self.network_bias_offsets[2],
            ],
            _padding2: 0,
        };
        unsafe {
            state.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push_constants),
            );
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

    fn destroy(&mut self, state: &mut VulkanState) {
        if let Some(allocation) = self.vertex_buffer_allocation.take() {
            state.allocator().free(allocation).unwrap();
        }
        if let Some(allocation) = self.network_buffer_allocation.take() {
            state.allocator().free(allocation).unwrap();
        }
        unsafe {
            state.device.destroy_buffer(self.vertex_buffer, None);
            state.device.destroy_buffer(self.network_buffer, None);
            state.device.destroy_pipeline(self.pipeline, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            state
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
