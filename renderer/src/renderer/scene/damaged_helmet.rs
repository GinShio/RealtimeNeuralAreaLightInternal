use anyhow::Result;
use ash::vk;
use crevice::std140::AsStd140;

use crate::renderer::{
    model_data::ModelData, render_images::RenderImages, scene::Scene, utils,
    vulkan_state::VulkanState,
};

#[repr(C)]
#[derive(AsStd140)]
struct PushConstants {
    model: glam::Mat4,
    view: glam::Mat4,
    projection: glam::Mat4,
}

pub struct DamagedHelmetScene {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    model_data: Vec<ModelData>,

    rotate: [f32; 3],
    camera_distance: f32,
    camera_rotate: [f32; 2],
}
impl DamagedHelmetScene {
    pub fn new(state: &mut VulkanState) -> Result<Box<Self>> {
        // Create graphics pipeline
        let (pipeline_layout, pipeline) = {
            utils::create_graphics_pipeline(
                state,
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/damaged_helmet.vert.spv"
                )),
                include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/shaders/scene/damaged_helmet.frag.spv"
                )),
                &[vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                    offset: 0,
                    size: std::mem::size_of::<Std140PushConstants>() as u32,
                }],
            )?
        };

        // Load model data
        let model_data = utils::load_glb(state, "./assets/DamagedHelmet.glb")?;

        Ok(Box::new(Self {
            pipeline_layout,
            pipeline,

            model_data,

            rotate: [0.0; 3],
            camera_distance: 2.5,
            camera_rotate: [0.0; 2],
        }))
    }
}
impl Scene for DamagedHelmetScene {
    fn scene_name(&self) -> &'static str {
        "DamagedHelmet Scene"
    }

    fn ui(&mut self, ui: &imgui::Ui) {
        let _id = ui.push_id("Model");
        ui.text("Model");
        imgui::Drag::new("rotate")
            .range(-180.0, 180.0)
            .build_array(ui, &mut self.rotate);

        ui.spacing();

        let _id = ui.push_id("Camera");
        ui.text("Camera");
        imgui::Drag::new("distance")
            .range(0.0, 10.0)
            .speed(0.1)
            .build(ui, &mut self.camera_distance);
        imgui::Drag::new("rotate")
            .range(-180.0, 180.0)
            .build_array(ui, &mut self.camera_rotate);
    }

    fn cmd_draw(
        &mut self,
        state: &VulkanState,
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

        for data in &self.model_data {
            // Bind vertex buffer
            let vertex_buffers = [data.vertex_buffer];
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
                    data.index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            // Push constants
            let model = glam::Mat4::from_euler(
                glam::EulerRot::YXZ,
                self.rotate[0].to_radians(),
                self.rotate[1].to_radians(),
                self.rotate[2].to_radians(),
            );
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
            let push_constants = PushConstants {
                model,
                view,
                projection,
            };
            unsafe {
                state.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    push_constants.as_std140().as_bytes(),
                );
            }

            // Draw Indexed
            unsafe {
                state
                    .device
                    .cmd_draw_indexed(command_buffer, data.index_count, 1, 0, 0, 0);
            }
        }

        // End rendering
        unsafe {
            state.device.cmd_end_rendering(command_buffer);
        }
    }

    fn destroy(&mut self, state: &mut VulkanState) {
        for data in &mut self.model_data {
            data.destroy(state);
        }
        unsafe {
            state.device.destroy_pipeline(self.pipeline, None);
            state
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
