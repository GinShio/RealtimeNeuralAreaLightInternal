use std::path::Path as FsPath;

use ash::vk;

use crate::{
    utils::{Texture, create_texture_with_mipmap_data},
    vulkan_state::VulkanState,
};

pub struct GlbTexture {
    pub base_color: Texture,
    pub metallic_roughness: Texture,
    pub normal: Texture,
}
impl GlbTexture {
    pub fn destroy(&mut self, state: &mut VulkanState) {
        self.base_color.destroy(state);
        self.metallic_roughness.destroy(state);
        self.normal.destroy(state);
    }
}

/// Load baseColor, metallic_roughness, and normal textures from a GLB file,
/// and return them with mipmaps.
pub fn load_glb_texture(
    state: &mut VulkanState,
    path: impl AsRef<FsPath>,
    width: u32,
) -> GlbTexture {
    // Load GLB using gltf::import
    let (document, _buffers, images) = gltf::import(path).expect("Failed to import GLB");

    // Get the first material from the GLB
    let material = document
        .materials()
        .next()
        .expect("No material found in GLB");

    // Get texture indices from the material
    let base_color_tex_index = material
        .pbr_metallic_roughness()
        .base_color_texture()
        .map(|info| info.texture().index());

    let metallic_roughness_tex_index = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .map(|info| info.texture().index());

    let normal_tex_index = material.normal_texture().map(|info| info.texture().index());

    // Get the first image from the GLB
    // baseColor
    let base_color = base_color_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| {
            let data = img
                .pixels
                .chunks(3)
                .flat_map(|p| {
                    let r = p[0];
                    let g = p[1];
                    let b = p[2];
                    vec![r, g, b, 255]
                })
                .collect::<Vec<u8>>();
            assert!(img.format == gltf::image::Format::R8G8B8);
            create_texture_with_mipmap_data(
                state,
                width,
                img.width,
                img.height,
                vk::Format::R8G8B8A8_UNORM,
                &data,
            )
        })
        .expect("No baseColor texture found");

    // metallic_roughness
    let metallic_roughness = metallic_roughness_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| {
            let data = img
                .pixels
                .chunks(3)
                .flat_map(|p| {
                    let r = p[0];
                    let g = p[1];
                    let b = p[2];
                    vec![r, g, b, 255]
                })
                .collect::<Vec<u8>>();
            assert!(img.format == gltf::image::Format::R8G8B8);
            create_texture_with_mipmap_data(
                state,
                width,
                img.width,
                img.height,
                vk::Format::R8G8B8A8_UNORM,
                &data,
            )
        })
        .expect("No metallic_roughness texture found");

    // normal
    let normal = normal_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| {
            let data = img
                .pixels
                .chunks(3)
                .flat_map(|p| {
                    let r = p[0];
                    let g = p[1];
                    let b = p[2];
                    vec![r, g, b, 255]
                })
                .collect::<Vec<u8>>();
            assert!(img.format == gltf::image::Format::R8G8B8);
            create_texture_with_mipmap_data(
                state,
                width,
                img.width,
                img.height,
                vk::Format::R8G8B8A8_UNORM,
                &data,
            )
        })
        .expect("No normal texture found");

    GlbTexture {
        base_color,
        metallic_roughness,
        normal,
    }
}
