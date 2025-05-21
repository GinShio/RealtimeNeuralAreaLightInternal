use ash::vk;

use crate::renderer::{
    render_images::RenderImages, texture_manager::TextureManager, vulkan_state::VulkanState,
};

mod damaged_helmet;
mod triangle;

pub use damaged_helmet::DamagedHelmetScene;
pub use triangle::TriangleScene;

pub trait Scene {
    fn scene_name(&self) -> &'static str;
    fn ui(&mut self, ui: &imgui::Ui);
    fn cmd_draw(
        &mut self,
        state: &VulkanState,
        texture_manager: &TextureManager,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
        render_images: &RenderImages,
    );
    fn destroy(&mut self, state: &mut VulkanState);
}
