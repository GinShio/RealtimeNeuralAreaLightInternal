use anyhow::Result;
use ash::vk;
use imgui::{Condition, Context, DrawData, Ui};
use imgui_rs_vulkan_renderer::{
    DynamicRendering as ImguiDynamicRendering, Options as ImguiOptions, Renderer as ImguiRenderer,
};

use crate::renderer::{
    Renderer, render_images::RenderImages, scene::Scene, vulkan_state::VulkanState,
};

/// A struct that represents the imgui pass.
pub struct ImGuiPass {
    imgui_renderer: Option<ImguiRenderer>,
}
impl ImGuiPass {
    /// Creates a new instance of the ImGuiPass struct.
    pub fn new(state: &VulkanState, imgui: &mut Context) -> Result<Self> {
        let imgui_renderer = ImguiRenderer::with_gpu_allocator(
            state.clone_allocator(),
            state.device.clone(),
            state.queue,
            state.command_pool,
            ImguiDynamicRendering {
                color_attachment_format: vk::Format::R8G8B8A8_UNORM,
                depth_attachment_format: None,
            },
            imgui,
            Some(ImguiOptions {
                in_flight_frames: Renderer::MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            }),
        )?;

        Ok(Self {
            imgui_renderer: Some(imgui_renderer),
        })
    }

    /// ImGui UI function.
    pub fn ui(
        &mut self,
        ui: &Ui,
        hidpi_factor: f32,
        render_time: f32,
        scene_index: &mut usize,
        scenes: &mut Vec<Box<dyn Scene>>,
    ) {
        let width = 250.0;
        let height = 300.0;
        let scene_names = scenes.iter().map(|s| s.scene_name()).collect::<Vec<_>>();
        let current_scene = &mut scenes[*scene_index];

        let w = ui
            .window("Scene Settings")
            .size([width, height], Condition::Appearing)
            .position(
                [1280.0 / hidpi_factor - width - 20.0, 20.0],
                Condition::Appearing,
            );

        w.build(|| {
            ui.text(format!("Render Time: {:.2} ms", render_time * 1000.0));
            ui.text(format!("FPS: {:.2}", 1.0 / render_time));
            ui.spacing();

            ui.combo_simple_string("Scene", scene_index, &scene_names);

            ui.spacing();
            ui.spacing();
            ui.separator();
            ui.spacing();
            ui.spacing();

            current_scene.ui(ui);
        });
    }

    /// Record the command buffer for the imgui pass.
    pub fn cmd_draw(
        &mut self,
        state: &VulkanState,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
        imgui_draw_data: &DrawData,
    ) {
        // Memory barrier
        // - after_tone_mapping_image[image_index] General -> ColorAttachmentOptimal
        let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2KHR::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(render_images.after_tone_mapping_images[image_index])
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
            .image_view(render_images.after_tone_mapping_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::LOAD)
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

        // Draw imgui
        self.imgui_renderer
            .as_mut()
            .unwrap()
            .cmd_draw(command_buffer, imgui_draw_data)
            .unwrap();

        // End rendering
        unsafe {
            state.device.cmd_end_rendering(command_buffer);
        }
    }

    /// Destroy the imgui pass.
    pub fn destroy(&mut self) {
        let imgui_renderer = self.imgui_renderer.take().unwrap();
        drop(imgui_renderer);
    }
}
