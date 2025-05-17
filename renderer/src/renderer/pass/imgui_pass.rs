use anyhow::Result;
use ash::vk;
use imgui::{Condition, Context, DrawData, Ui};
use imgui_rs_vulkan_renderer::{
    DynamicRendering as ImguiDynamicRendering, Options as ImguiOptions, Renderer as ImguiRenderer,
};

use crate::renderer::{Renderer, render_images::RenderImages, vulkan_state::VulkanState};

/// A struct that represents the imgui pass.
pub struct ImGuiPass {
    imgui_renderer: Option<ImguiRenderer>,
    notify_text: &'static str,
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

        let notify_text = "Hello from ImGui!";

        Ok(Self {
            imgui_renderer: Some(imgui_renderer),
            notify_text,
        })
    }

    /// ImGui UI function.
    pub fn ui(&mut self, ui: &Ui, hidpi_factor: f32) {
        let width = 300.0;
        let height = 200.0;
        let w = ui
            .window("ImGui Color Button Example")
            .size([width, height], Condition::Appearing)
            .position(
                [1280.0 / hidpi_factor - width - 20.0, 20.0],
                Condition::Appearing,
            );
        w.build(|| {
            ui.text_wrapped(
                "Color button is a widget that displays a color value as a clickable rectangle. \
             It also supports a tooltip with detailed information about the color value. \
             Try hovering over and clicking these buttons!",
            );
            ui.text(self.notify_text);

            ui.text("This button is black:");
            if ui.color_button("Black color", [0.0, 0.0, 0.0, 1.0]) {
                self.notify_text = "*** Black button was clicked";
            }

            ui.text("This button is red:");
            if ui.color_button("Red color", [1.0, 0.0, 0.0, 1.0]) {
                self.notify_text = "*** Red button was clicked";
            }

            ui.text("This button is BIG because it has a custom size:");
            if ui
                .color_button_config("Green color", [0.0, 1.0, 0.0, 1.0])
                .size([100.0, 50.0])
                .build()
            {
                self.notify_text = "*** BIG button was clicked";
            }

            ui.text("This button doesn't use the tooltip at all:");
            if ui
                .color_button_config("No tooltip", [0.0, 0.0, 1.0, 1.0])
                .tooltip(false)
                .build()
            {
                self.notify_text = "*** No tooltip button was clicked";
            }
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
