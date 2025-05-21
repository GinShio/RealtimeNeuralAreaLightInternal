use ash::vk;

use crate::renderer::scene::Scene;
use crate::renderer::{
    render_images::RenderImages, texture_manager::TextureManager, vulkan_state::VulkanState,
};

/// A struct that represents the scene pass.
pub struct ScenePass;
impl ScenePass {
    /// Creates a new instance of the ScenePass struct.
    pub fn new() -> Self {
        Self
    }

    /// Record the command buffer for the scene pass.
    pub fn cmd_draw(
        &self,
        state: &VulkanState,
        texture_manager: &TextureManager,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
        scene: &mut Box<dyn Scene>,
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

        // Record the command buffer
        scene.cmd_draw(
            state,
            texture_manager,
            command_buffer,
            image_index,
            render_images,
        );
    }

    /// Destroy the scene pass.
    pub fn destroy(&mut self) {
        // do nothing
    }
}
