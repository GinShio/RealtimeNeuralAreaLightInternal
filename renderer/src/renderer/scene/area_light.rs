use std::path::Path;

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
    rotation: [f32; 3],
    size: f32,
    color: [f32; 3],
    aspect_ratio: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBuffer {
    view: [f32; 16],
    projection: [f32; 16],

    sampler: SamplerIndex,

    light_count: u32,

    _padding: [u32; 2],

    lights: [LightParams; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelPushConstants {
    model: [f32; 16],
    weight_offsets_0: [u32; 4],
    weight_offsets_1: [u32; 4],
    bias_offsets_0: [u32; 4],
    bias_offsets_1: [u32; 4],
    latent_texture_0: TextureIndex,
    latent_texture_1: TextureIndex,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightPushConstants {
    light_index: u32,
}

struct NetworkModel {
    model_data: Vec<ModelData>,
    latent_texture_0: TextureIndex,
    latent_texture_1: TextureIndex,
    network_buffer: vk::Buffer,
    network_buffer_allocation: Option<Allocation>,
    network_weight_offsets: Vec<u32>,
    network_bias_offsets: Vec<u32>,
    network_descriptor_set: vk::DescriptorSet,
}
impl NetworkModel {
    fn new(
        state: &mut VulkanState,
        texture_manager: &mut TextureManager,
        model_path: impl AsRef<Path>,
        network_json_path: impl AsRef<Path>,
        latent_texture_0_path: impl AsRef<Path>,
        latent_texture_1_path: impl AsRef<Path>,
        network_descriptor_pool: vk::DescriptorPool,
        network_descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        // Load glb
        let model_data = utils::load_glb_without_texture(state, model_path.as_ref())?;

        // Load latent textures
        let latent_texture_0 =
            texture_manager.load_latent_texture(state, latent_texture_0_path.as_ref())?;
        let latent_texture_1 =
            texture_manager.load_latent_texture(state, latent_texture_1_path.as_ref())?;

        // Load network data
        let network = Network::from_json(&state.cooperative_vector_fn, network_json_path.as_ref())?;
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

            // Return the network buffer and its memory
            (network_buffer, network_buffer_allocation)
        };

        let network_descriptor_set = {
            let set_layouts = [network_descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(network_descriptor_pool)
                .set_layouts(&set_layouts);
            let descriptor_sets = unsafe { state.device.allocate_descriptor_sets(&allocate_info)? };
            descriptor_sets[0]
        };
        {
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
        }

        Ok(Self {
            model_data,
            latent_texture_0,
            latent_texture_1,
            network_buffer,
            network_buffer_allocation: Some(network_buffer_allocation),
            network_weight_offsets: network.weight_offsets,
            network_bias_offsets: network.bias_offsets,
            network_descriptor_set,
        })
    }

    fn destroy(&mut self, state: &mut VulkanState) {
        if let Some(allocation) = self.network_buffer_allocation.take() {
            state.allocator().free(allocation).unwrap();
        }
        unsafe {
            state.device.destroy_buffer(self.network_buffer, None);
        }
        for model in &mut self.model_data {
            model.destroy(state);
        }
    }
}

struct LightData {
    position: [f32; 3],
    intensity: f32,
    rotation: [f32; 3],
    size: f32,
    color: [f32; 3],
    aspect_ratio: f32,
}

pub struct AreaLightScene {
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_allocations: Vec<Allocation>,
    uniform_descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_descriptor_pool: vk::DescriptorPool,
    uniform_descriptor_sets: Vec<vk::DescriptorSet>,

    pipeline_layout_mlp: vk::PipelineLayout,
    pipeline_mlp: vk::Pipeline,

    pipeline_layout_light: vk::PipelineLayout,
    pipeline_light: vk::Pipeline,

    sampler: SamplerIndex,

    camera_distance: f32,
    camera_rotate: [f32; 2],
    camera_pivot: [f32; 3],

    network_descriptor_set_layout: vk::DescriptorSetLayout,
    network_descriptor_pool: vk::DescriptorPool,
    network_models: Vec<NetworkModel>,

    light_count: usize,
    lights: Vec<LightData>,
    light_model: Vec<ModelData>,

    model_set_index: usize,
}
impl AreaLightScene {
    pub fn new(state: &mut VulkanState, texture_manager: &mut TextureManager) -> Result<Box<Self>> {
        // Create uniform buffer
        let (uniform_buffers, uniform_buffer_allocations) = (0..Renderer::IMAGE_COUNT)
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
                .descriptor_count(Renderer::IMAGE_COUNT as u32)];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(Renderer::IMAGE_COUNT as u32);
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
                .take(Renderer::IMAGE_COUNT)
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
                .max_sets(128);
            unsafe {
                state
                    .device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)?
            }
        };

        // Load network models
        let network_models = vec![NetworkModel::new(
            state,
            texture_manager,
            "assets/pbr-simple/plane/plane.glb",
            "network/pbr-simple/pre/network.json",
            "network/pbr-simple/pre/latent-texture-0.exr",
            "network/pbr-simple/pre/latent-texture-1.exr",
            network_descriptor_pool,
            network_descriptor_set_layout,
        )?];

        // Create graphics pipeline
        let (pipeline_layout_mlp, pipeline_mlp) = {
            utils::create_graphics_pipeline(
                state,
                texture_manager,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/area_light/mlp.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/area_light/mlp.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: std::mem::size_of::<ModelPushConstants>() as u32,
                }],
                &[uniform_descriptor_set_layout, network_descriptor_set_layout],
                true,
            )?
        };
        let (pipeline_layout_light, pipeline_light) = {
            utils::create_graphics_pipeline(
                state,
                texture_manager,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/area_light/light.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/area_light/light.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    offset: 0,
                    size: std::mem::size_of::<LightPushConstants>() as u32,
                }],
                &[uniform_descriptor_set_layout],
                false,
            )?
        };

        // Create sampler
        let sampler = texture_manager.create_sampler(
            state,
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR),
        )?;

        // Create Light Data
        let light_count = 0;
        let lights = vec![
            LightData {
                position: [0.0, 1.0, -5.0],
                intensity: 10.0,
                rotation: [0.0, 0.0, 0.0],
                size: 1.0,
                color: [1.0, 1.0, 1.0],
                aspect_ratio: 1.0,
            },
            LightData {
                position: [5.0, 1.0, -5.0],
                intensity: 10.0,
                rotation: [0.0, 0.0, 0.0],
                size: 1.0,
                color: [1.0, 1.0, 1.0],
                aspect_ratio: 1.0,
            },
            LightData {
                position: [-5.0, 1.0, -5.0],
                intensity: 10.0,
                rotation: [0.0, 0.0, 0.0],
                size: 1.0,
                color: [1.0, 1.0, 1.0],
                aspect_ratio: 1.0,
            },
        ];
        let light_model =
            utils::load_glb_without_texture(state, "assets/area-light/light-rectangle.glb")?;

        Ok(Box::new(Self {
            uniform_buffers,
            uniform_buffer_allocations,
            uniform_descriptor_set_layout,
            uniform_descriptor_pool,
            uniform_descriptor_sets,

            pipeline_layout_mlp,
            pipeline_mlp,

            pipeline_layout_light,
            pipeline_light,

            sampler,

            camera_distance: 10.0,
            camera_rotate: [-30.0, 0.0],
            camera_pivot: [0.0, 1.0, 0.0],

            network_descriptor_set_layout,
            network_descriptor_pool,
            network_models,

            light_count,
            lights,
            light_model,

            model_set_index: 0,
        }))
    }
}
impl Scene for AreaLightScene {
    fn scene_name(&self) -> &'static str {
        "Area Light Scene"
    }

    fn ui(&mut self, ui: &imgui::Ui) {
        ui.text("Area Light Scene");
        ui.separator();

        ui.combo_simple_string(
            "model set",
            &mut self.model_set_index,
            &["pbr-simple plane"],
        );

        ui.text("Camera");
        imgui::Drag::new("distance")
            .range(0.0, 10.0)
            .speed(0.01)
            .build(ui, &mut self.camera_distance);
        imgui::Drag::new("rotate")
            .range(-180.0, 180.0)
            .build_array(ui, &mut self.camera_rotate);
        imgui::Drag::new("pivot")
            .range(-10.0, 10.0)
            .speed(0.01)
            .build_array(ui, &mut self.camera_pivot);

        ui.spacing();
        ui.separator();
        ui.spacing();

        ui.text("Lights");

        ui.combo_simple_string("light count", &mut self.light_count, &["1", "2", "3"]);

        for i in 0..(self.light_count + 1) as usize {
            let light = &mut self.lights[i];
            ui.text(format!("Light {}", i + 1));
            imgui::Drag::new("position")
                .range(-10.0, 10.0)
                .speed(0.01)
                .build_array(ui, &mut light.position);
            imgui::Drag::new("size")
                .range(0.1, 5.0)
                .speed(0.01)
                .build(ui, &mut light.size);
            imgui::Drag::new("aspect ratio")
                .range(1.0 / 16.0, 16.0)
                .speed(0.01)
                .build(ui, &mut light.aspect_ratio);
            imgui::Drag::new("rotation")
                .range(-180.0, 180.0)
                .build_array(ui, &mut light.rotation);
            imgui::Drag::new("intensity")
                .range(0.0, 100.0)
                .speed(0.1)
                .build(ui, &mut light.intensity);
            imgui::Drag::new("color")
                .range(0.0, 1.0)
                .speed(0.01)
                .build_array(ui, &mut light.color);
            ui.separator();
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
                    float32: [0.0, 0.0, 0.0, 1.0],
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

        // Create uniform buffer
        let camera_position = glam::Mat3::from_euler(
            glam::EulerRot::YXZ,
            self.camera_rotate[1].to_radians(),
            self.camera_rotate[0].to_radians(),
            0.0,
        ) * glam::Vec3::new(0.0, 0.0, self.camera_distance)
            + glam::Vec3::from(self.camera_pivot);
        let view = glam::Mat4::look_at_rh(
            camera_position,
            glam::Vec3::from(self.camera_pivot),
            glam::Vec3::Y,
        );
        let projection = glam::Mat4::perspective_rh(
            60.0_f32.to_radians(),
            state.swapchain.extent.width as f32 / state.swapchain.extent.height as f32,
            0.01,
            100.0,
        );
        let lights = [
            LightParams {
                position: self.lights[0].position,
                intensity: self.lights[0].intensity,
                rotation: [
                    self.lights[0].rotation[0].to_radians(),
                    self.lights[0].rotation[1].to_radians(),
                    self.lights[0].rotation[2].to_radians(),
                ],
                size: self.lights[0].size,
                color: self.lights[0].color,
                aspect_ratio: self.lights[0].aspect_ratio,
            },
            LightParams {
                position: self.lights[1].position,
                intensity: self.lights[1].intensity,
                rotation: [
                    self.lights[1].rotation[0].to_radians(),
                    self.lights[1].rotation[1].to_radians(),
                    self.lights[1].rotation[2].to_radians(),
                ],
                size: self.lights[1].size,
                color: self.lights[1].color,
                aspect_ratio: self.lights[1].aspect_ratio,
            },
            LightParams {
                position: self.lights[2].position,
                intensity: self.lights[2].intensity,
                rotation: [
                    self.lights[2].rotation[0].to_radians(),
                    self.lights[2].rotation[1].to_radians(),
                    self.lights[2].rotation[2].to_radians(),
                ],
                size: self.lights[2].size,
                color: self.lights[2].color,
                aspect_ratio: self.lights[2].aspect_ratio,
            },
        ];
        let uniform_buffer = UniformBuffer {
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

        // Bind mlp pipeline
        unsafe {
            state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_mlp,
            );
        }

        // draw models
        let mut models = vec![];
        match self.model_set_index {
            0 => {
                // pbr-simple plane
                models.push((&self.network_models[0], glam::Mat4::IDENTITY));
            }
            _ => panic!("Unknown model set index"),
        }
        for (model, model_matrix) in models {
            // Bind descriptor sets
            let mut descriptor_sets = vec![];
            descriptor_sets.extend(texture_manager.descriptor_sets());
            descriptor_sets.push(self.uniform_descriptor_sets[image_index]);
            descriptor_sets.push(model.network_descriptor_set);
            unsafe {
                state.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout_mlp,
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

            // Push constants
            let push_constants = ModelPushConstants {
                model: model_matrix.to_cols_array(),
                weight_offsets_0: [
                    model.network_weight_offsets[0],
                    model.network_weight_offsets[1],
                    model.network_weight_offsets[2],
                    model.network_weight_offsets[3],
                ],
                weight_offsets_1: [
                    model.network_weight_offsets[4],
                    model.network_weight_offsets[5],
                    model.network_weight_offsets[6],
                    model.network_weight_offsets[7],
                    // 0,
                    // 0,
                    // 0,
                ],
                bias_offsets_0: [
                    model.network_bias_offsets[0],
                    model.network_bias_offsets[1],
                    model.network_bias_offsets[2],
                    model.network_bias_offsets[3],
                ],
                bias_offsets_1: [
                    model.network_bias_offsets[4],
                    model.network_bias_offsets[5],
                    model.network_bias_offsets[6],
                    model.network_bias_offsets[7],
                    // 0,
                    // 0,
                    // 0,
                ],
                latent_texture_0: model.latent_texture_0,
                latent_texture_1: model.latent_texture_1,
            };
            unsafe {
                state.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout_mlp,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
            }

            // Draw models
            for model_data in &model.model_data {
                let vertex_buffers = [model_data.vertex_buffer];
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
                        model_data.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                // Draw Indexed
                unsafe {
                    state.device.cmd_draw_indexed(
                        command_buffer,
                        model_data.index_count,
                        1,
                        0,
                        0,
                        0,
                    );
                }
            }
        }

        // Bind light pipeline
        unsafe {
            state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_light,
            );
        }

        for i in 0..(self.light_count + 1) {
            // Bind descriptor sets
            let mut descriptor_sets = vec![];
            descriptor_sets.extend(texture_manager.descriptor_sets());
            descriptor_sets.push(self.uniform_descriptor_sets[image_index]);
            unsafe {
                state.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout_light,
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

            // Push constants
            let push_constants = LightPushConstants {
                light_index: i as u32,
            };
            unsafe {
                state.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout_light,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
            }

            for model_data in &self.light_model {
                // Bind vertex buffers
                let vertex_buffers = [model_data.vertex_buffer];
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
                        model_data.index_buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }

                // Draw Indexed
                unsafe {
                    state.device.cmd_draw_indexed(
                        command_buffer,
                        model_data.index_count,
                        1,
                        0,
                        0,
                        0,
                    );
                }
            }
        }

        // End rendering
        unsafe {
            state.device.cmd_end_rendering(command_buffer);
        }
    }

    fn destroy(&mut self, state: &mut VulkanState) {
        for allocation in self.uniform_buffer_allocations.drain(..) {
            state.allocator().free(allocation).unwrap();
        }

        for model_data in &mut self.light_model {
            model_data.destroy(state);
        }

        unsafe {
            for uniform_buffer in self.uniform_buffers.drain(..) {
                state.device.destroy_buffer(uniform_buffer, None);
            }
            state
                .device
                .destroy_descriptor_pool(self.uniform_descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.uniform_descriptor_set_layout, None);

            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout_mlp, None);
            state.device.destroy_pipeline(self.pipeline_mlp, None);

            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout_light, None);
            state.device.destroy_pipeline(self.pipeline_light, None);

            state
                .device
                .destroy_descriptor_pool(self.network_descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.network_descriptor_set_layout, None);
        }
        for model in &mut self.network_models {
            model.destroy(state);
        }
    }
}
