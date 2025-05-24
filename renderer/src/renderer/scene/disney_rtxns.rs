use anyhow::Result;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::renderer::{
    Renderer,
    model_data::ModelData,
    network::Network,
    render_images::RenderImages,
    scene::Scene,
    texture_manager::{SamplerIndex, TextureIndex, TextureManager},
    utils,
    vulkan_state::VulkanState,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightParams {
    position: [f32; 3],
    intensity: f32,
    color: [f32; 3],
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBuffer {
    weight_offsets: [u32; 4],
    bias_offsets: [u32; 4],

    view: [f32; 16],
    projection: [f32; 16],

    sampler: SamplerIndex,

    light_count: u32,

    _padding: [u32; 2],

    lights: [LightParams; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    model: [f32; 16],
    base_color: TextureIndex,
    normal: TextureIndex,
    metallic_roughness: TextureIndex,
    _padding: u32,
    base_color_override_value: [f32; 3],
    base_color_override: u32,
    roughness_override_value: f32,
    roughness_override: u32,
    metallic_override_value: f32,
    metallic_override: u32,
}

pub struct DisneyRtxnsScene {
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    uniform_descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_descriptor_pool: vk::DescriptorPool,
    uniform_descriptor_sets: Vec<vk::DescriptorSet>,

    network_descriptor_set_layout: vk::DescriptorSetLayout,
    network_descriptor_pool: vk::DescriptorPool,
    network_descriptor_set: vk::DescriptorSet,

    pipeline_layout_mlp: vk::PipelineLayout,
    pipeline_mlp: vk::Pipeline,

    pipeline_layout_analytic: vk::PipelineLayout,
    pipeline_analytic: vk::Pipeline,

    pipeline_layout_diff: vk::PipelineLayout,
    pipeline_diff: vk::Pipeline,

    sampler: SamplerIndex,
    helmet_model_data: Vec<(ModelData, utils::GltfTextures)>,
    sphere_model_data: ModelData,

    network_buffer: vk::Buffer,
    network_buffer_allocation: Option<Allocation>,
    network_weight_offsets: Vec<u32>,
    network_bias_offsets: Vec<u32>,

    camera_distance: f32,
    camera_rotate: [f32; 2],

    light_count: usize,

    light0_position: [f32; 3],
    light0_intensity: f32,
    light0_color: [f32; 3],

    light1_position: [f32; 3],
    light1_intensity: f32,
    light1_color: [f32; 3],

    light2_position: [f32; 3],
    light2_intensity: f32,
    light2_color: [f32; 3],

    base_color_override_value: [f32; 3],
    base_color_override: bool,
    roughness_override_value: f32,
    roughness_override: bool,
    metallic_override_value: f32,
    metallic_override: bool,

    pipeline_index: usize,
    model_index: usize,
}
impl DisneyRtxnsScene {
    pub fn new(state: &mut VulkanState, texture_manager: &mut TextureManager) -> Result<Box<Self>> {
        // Create uniform buffer
        let (uniform_buffers, uniform_buffer_allocations) = (0..Renderer::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                // Create uniform buffer
                let uniform_buffer_create_info = vk::BufferCreateInfo::default()
                    .size(std::mem::size_of::<UniformBuffer>() as u64)
                    .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let uniform_buffer = unsafe {
                    state
                        .device
                        .create_buffer(&uniform_buffer_create_info, None)
                        .unwrap()
                };
                // Allocate memory for the uniform buffer
                let uniform_buffer_requirements =
                    unsafe { state.device.get_buffer_memory_requirements(uniform_buffer) };
                let uniform_buffer_allocation = state
                    .allocator()
                    .allocate(&AllocationCreateDesc {
                        name: "uniform buffer",
                        requirements: uniform_buffer_requirements,
                        location: MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();
                // Bind the uniform buffer memory
                unsafe {
                    state
                        .device
                        .bind_buffer_memory(
                            uniform_buffer,
                            uniform_buffer_allocation.memory(),
                            uniform_buffer_allocation.offset(),
                        )
                        .unwrap();
                }
                (uniform_buffer, uniform_buffer_allocation)
            })
            .collect::<(Vec<_>, Vec<_>)>();
        // Create uniform descriptor set layout
        let uniform_descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
            let descriptor_set_layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe {
                state
                    .device
                    .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?
            }
        };
        // Create uniform descriptor pool
        let uniform_descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(Renderer::MAX_FRAMES_IN_FLIGHT as u32)];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(Renderer::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe {
                state
                    .device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)?
            }
        };
        // Create uniform descriptor sets
        let uniform_descriptor_sets = {
            let set_layouts = [uniform_descriptor_set_layout]
                .into_iter()
                .cycle()
                .take(Renderer::MAX_FRAMES_IN_FLIGHT)
                .collect::<Vec<_>>();
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(uniform_descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe {
                state
                    .device
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)?
            }
        };
        // Bind uniform buffer to descriptor sets
        for (i, uniform_buffer) in uniform_buffers.iter().enumerate() {
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(*uniform_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let write_descriptor_set = vk::WriteDescriptorSet::default()
                .dst_set(uniform_descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info);
            unsafe {
                state
                    .device
                    .update_descriptor_sets(&[write_descriptor_set], &[])
            };
        }

        // Create network descriptor set layout
        let network_descriptor_set_layout = {
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
        // Create network descriptor pool
        let network_descriptor_pool = {
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
        // Create network descriptor set
        let network_descriptor_set = {
            let set_layouts = [network_descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(network_descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_sets = unsafe { state.device.allocate_descriptor_sets(&allocate_info)? };
            descriptor_sets[0]
        };

        // Create graphics pipeline
        let (pipeline_layout_mlp, pipeline_mlp) = {
            utils::create_graphics_pipeline(
                state,
                texture_manager,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/disney_rtxns/mlp.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/disney_rtxns/mlp.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: std::mem::size_of::<PushConstants>() as u32,
                }],
                &[network_descriptor_set_layout, uniform_descriptor_set_layout],
            )?
        };
        let (pipeline_layout_analytic, pipeline_analytic) = {
            utils::create_graphics_pipeline(
                state,
                texture_manager,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/disney_rtxns/analytic.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/disney_rtxns/analytic.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: std::mem::size_of::<PushConstants>() as u32,
                }],
                &[network_descriptor_set_layout, uniform_descriptor_set_layout],
            )?
        };
        let (pipeline_layout_diff, pipeline_diff) = {
            utils::create_graphics_pipeline(
                state,
                texture_manager,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/disney_rtxns/diff.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/disney_rtxns/diff.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: std::mem::size_of::<PushConstants>() as u32,
                }],
                &[network_descriptor_set_layout, uniform_descriptor_set_layout],
            )?
        };

        // Create sampler
        let sampler = texture_manager.create_sampler(
            state,
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR),
        )?;

        // Load model data
        let helmet_model_data =
            utils::load_glb(state, texture_manager, "./assets/DamagedHelmet.glb")?;
        let sphere_model_data = utils::load_sphere(state, 128, 128);

        // Load network data
        let network =
            Network::from_json(&state.cooperative_vector_fn, "./network/disney-rtxns.json")?;
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

        // Update network descriptor set
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(network_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info);
        unsafe {
            state
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[])
        };

        Ok(Box::new(Self {
            uniform_buffers,
            uniform_buffer_allocations,
            uniform_descriptor_set_layout,
            uniform_descriptor_pool,
            uniform_descriptor_sets,

            network_descriptor_set_layout,
            network_descriptor_pool,
            network_descriptor_set,

            pipeline_layout_mlp,
            pipeline_mlp,

            pipeline_layout_analytic,
            pipeline_analytic,

            pipeline_layout_diff,
            pipeline_diff,

            sampler,
            helmet_model_data,
            sphere_model_data,

            network_buffer,
            network_buffer_allocation: Some(network_buffer_allocation),
            network_weight_offsets: network.weight_offsets,
            network_bias_offsets: network.bias_offsets,

            camera_distance: 2.5,
            camera_rotate: [0.0; 2],

            light_count: 0,

            light0_position: [-2.5, 2.5, 2.5],
            light0_intensity: 10.0,
            light0_color: [1.0; 3],

            light1_position: [2.5, 2.5, 2.5],
            light1_intensity: 10.0,
            light1_color: [1.0; 3],

            light2_position: [0.0, 2.5, 2.5],
            light2_intensity: 10.0,
            light2_color: [1.0; 3],

            base_color_override_value: [1.0; 3],
            base_color_override: false,
            roughness_override_value: 0.5,
            roughness_override: false,
            metallic_override_value: 0.0,
            metallic_override: false,

            pipeline_index: 0,
            model_index: 0,
        }))
    }
}
impl Scene for DisneyRtxnsScene {
    fn scene_name(&self) -> &'static str {
        "Disney RTXNS"
    }

    fn ui(&mut self, ui: &imgui::Ui) {
        ui.combo_simple_string(
            "render type",
            &mut self.pipeline_index,
            &["MLP", "Analytic", "Diff"],
        );

        ui.combo_simple_string("model type", &mut self.model_index, &["Helmet", "Sphere"]);

        ui.text("Camera");
        imgui::Drag::new("distance")
            .range(0.0, 10.0)
            .speed(0.1)
            .build(ui, &mut self.camera_distance);
        imgui::Drag::new("rotate")
            .range(-180.0, 180.0)
            .build_array(ui, &mut self.camera_rotate);

        ui.text("Material Override");
        ui.checkbox("base color override", &mut self.base_color_override);
        if self.base_color_override {
            ui.color_edit3("base color", &mut self.base_color_override_value);
        }
        ui.checkbox("roughness override", &mut self.roughness_override);
        if self.roughness_override {
            imgui::Drag::new("roughness")
                .range(0.0, 1.0)
                .speed(0.01)
                .build(ui, &mut self.roughness_override_value);
        }
        ui.checkbox("metallic override", &mut self.metallic_override);
        if self.metallic_override {
            imgui::Drag::new("metallic")
                .range(0.0, 1.0)
                .speed(0.01)
                .build(ui, &mut self.metallic_override_value);
        }

        ui.text("Lights");

        ui.combo_simple_string("light count", &mut self.light_count, &["1", "2", "3"]);

        imgui::Drag::new("light0 position")
            .range(-10.0, 10.0)
            .build_array(ui, &mut self.light0_position);
        imgui::Drag::new("light0 intensity")
            .range(0.0, 100.0)
            .speed(0.01)
            .build(ui, &mut self.light0_intensity);
        ui.color_edit3("light0 color", &mut self.light0_color);

        if self.light_count >= 1 {
            imgui::Drag::new("light1 position")
                .range(-10.0, 10.0)
                .build_array(ui, &mut self.light1_position);
            imgui::Drag::new("light1 intensity")
                .range(0.0, 100.0)
                .speed(0.01)
                .build(ui, &mut self.light1_intensity);
            ui.color_edit3("light1 color", &mut self.light1_color);
        }
        if self.light_count >= 2 {
            imgui::Drag::new("light2 position")
                .range(-10.0, 10.0)
                .build_array(ui, &mut self.light2_position);
            imgui::Drag::new("light2 intensity")
                .range(0.0, 100.0)
                .speed(0.01)
                .build(ui, &mut self.light2_intensity);
            ui.color_edit3("light2 color", &mut self.light2_color);
        }
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
        match self.pipeline_index {
            0 => unsafe {
                state.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_mlp,
                );
            },
            1 => unsafe {
                state.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_analytic,
                );
            },
            2 => unsafe {
                state.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_diff,
                );
            },
            _ => unreachable!(),
        }

        // Uniform buffer
        let camera_position = glam::Mat3::from_euler(
            glam::EulerRot::YXZ,
            self.camera_rotate[0].to_radians(),
            self.camera_rotate[1].to_radians(),
            0.0,
        ) * glam::Vec3::new(0.0, 0.0, self.camera_distance);
        let view = glam::Mat4::look_at_rh(camera_position, glam::Vec3::ZERO, glam::Vec3::Y);
        let projection = glam::Mat4::perspective_rh(
            60.0_f32.to_radians(),
            state.swapchain.extent.width as f32 / state.swapchain.extent.height as f32,
            0.01,
            100.0,
        );
        let lights = [
            LightParams {
                position: self.light0_position,
                intensity: self.light0_intensity,
                color: self.light0_color,
                _padding: 0,
            },
            LightParams {
                position: self.light1_position,
                intensity: self.light1_intensity,
                color: self.light1_color,
                _padding: 0,
            },
            LightParams {
                position: self.light2_position,
                intensity: self.light2_intensity,
                color: self.light2_color,
                _padding: 0,
            },
        ];
        let uniform_buffer = UniformBuffer {
            weight_offsets: [
                self.network_weight_offsets[0],
                self.network_weight_offsets[1],
                self.network_weight_offsets[2],
                self.network_weight_offsets[3],
            ],
            bias_offsets: [
                self.network_bias_offsets[0],
                self.network_bias_offsets[1],
                self.network_bias_offsets[2],
                self.network_bias_offsets[3],
            ],

            view: view.to_cols_array(),
            projection: projection.to_cols_array(),

            sampler: self.sampler,

            light_count: (self.light_count + 1) as u32,

            _padding: [0; 2],

            lights,
        };

        // Update uniform buffer
        {
            let uniform_buffer_allocation = &mut self.uniform_buffer_allocations[image_index];
            let data = uniform_buffer_allocation
                .mapped_slice_mut()
                .expect("Failed to map uniform buffer memory");
            data[0..std::mem::size_of::<UniformBuffer>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform_buffer));
        }

        // Bind descriptor sets
        let pipeline_layout = match self.pipeline_index {
            0 => self.pipeline_layout_mlp,
            1 => self.pipeline_layout_analytic,
            2 => self.pipeline_layout_diff,
            _ => unreachable!(),
        };
        let mut descriptor_sets = vec![];
        descriptor_sets.extend(texture_manager.descriptor_sets());
        descriptor_sets.push(self.network_descriptor_set);
        descriptor_sets.push(self.uniform_descriptor_sets[image_index]);
        unsafe {
            state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
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

        if self.model_index == 0 {
            for (data, texture) in &self.helmet_model_data {
                // Bind vertex buffer
                let vertex_buffers = [data.vertex_buffer];
                let offsets = [0];
                unsafe {
                    state.device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &vertex_buffers,
                        &offsets,
                    );
                }

                // Bind index buffer
                unsafe {
                    state.device.cmd_bind_index_buffer(
                        command_buffer,
                        data.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                // Push constants
                let model = glam::Mat4::IDENTITY;
                let push_constants = PushConstants {
                    model: model.to_cols_array(),
                    base_color: texture.base_color.unwrap_or(TextureIndex::invalid()),
                    normal: texture.normal.unwrap_or(TextureIndex::invalid()),
                    metallic_roughness: texture
                        .metallic_roughness
                        .unwrap_or(TextureIndex::invalid()),
                    _padding: 0,
                    base_color_override_value: self.base_color_override_value,
                    base_color_override: self.base_color_override as u32,
                    roughness_override_value: self.roughness_override_value,
                    roughness_override: self.roughness_override as u32,
                    metallic_override_value: self.metallic_override_value,
                    metallic_override: self.metallic_override as u32,
                };
                unsafe {
                    state.device.cmd_push_constants(
                        command_buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push_constants),
                    );
                }

                // Draw Indexed
                unsafe {
                    state
                        .device
                        .cmd_draw_indexed(command_buffer, data.index_count, 1, 0, 0, 0);
                }
            }
        } else if self.model_index == 1 {
            // Bind vertex buffer
            let vertex_buffers = [self.sphere_model_data.vertex_buffer];
            let offsets = [0];
            unsafe {
                state
                    .device
                    .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            }

            // Bind index buffer
            unsafe {
                state.device.cmd_bind_index_buffer(
                    command_buffer,
                    self.sphere_model_data.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            // Push constants
            let model = glam::Mat4::IDENTITY;
            let push_constants = PushConstants {
                model: model.to_cols_array(),
                base_color: TextureIndex::invalid(),
                normal: TextureIndex::invalid(),
                metallic_roughness: TextureIndex::invalid(),
                _padding: 0,
                base_color_override_value: self.base_color_override_value,
                base_color_override: self.base_color_override as u32,
                roughness_override_value: self.roughness_override_value,
                roughness_override: self.roughness_override as u32,
                metallic_override_value: self.metallic_override_value,
                metallic_override: self.metallic_override as u32,
            };
            unsafe {
                state.device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
            }

            // Draw Indexed
            unsafe {
                state.device.cmd_draw_indexed(
                    command_buffer,
                    self.sphere_model_data.index_count,
                    1,
                    0,
                    0,
                    0,
                );
            }
        }

        // End rendering
        unsafe {
            state.device.cmd_end_rendering(command_buffer);
        }
    }

    fn destroy(&mut self, state: &mut VulkanState) {
        if let Some(allocation) = self.network_buffer_allocation.take() {
            state.allocator().free(allocation).unwrap();
        }

        for (data, _) in &mut self.helmet_model_data {
            data.destroy(state);
        }
        self.sphere_model_data.destroy(state);

        for allocation in self.uniform_buffer_allocations.drain(..) {
            state.allocator().free(allocation).unwrap();
        }

        unsafe {
            for uniform_buffer in &self.uniform_buffers {
                state.device.destroy_buffer(*uniform_buffer, None);
            }

            state.device.destroy_buffer(self.network_buffer, None);

            state.device.destroy_pipeline(self.pipeline_mlp, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout_mlp, None);

            state.device.destroy_pipeline(self.pipeline_analytic, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout_analytic, None);

            state.device.destroy_pipeline(self.pipeline_diff, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout_diff, None);

            state
                .device
                .destroy_descriptor_pool(self.uniform_descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.uniform_descriptor_set_layout, None);

            state
                .device
                .destroy_descriptor_pool(self.network_descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.network_descriptor_set_layout, None);
        }
    }
}
