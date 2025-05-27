use std::path::Path as FsPath;

use gltf::image::Data as GltfImageData;

use crate::{
    utils::{Texture, create_texture_with_mipmap},
    vulkan_state::VulkanState,
};

pub struct GlbTexture {
    pub base_color: Texture,
    pub metallic_roughness: Option<Texture>,
    pub normal: Option<Texture>,
}

/// GLBファイルからbaseColor, metallic_roughness, normalのテクスチャを読み込み、
/// mipmap付きで返す
pub fn load_glb_texture(
    state: &mut VulkanState,
    path: impl AsRef<FsPath>,
    width: u32,
) -> GlbTexture {
    // gltf::importでGLBを読み込む
    let (document, _buffers, images) = gltf::import(path).expect("Failed to import GLB");

    // マテリアルからテクスチャindexを取得
    let material = document
        .materials()
        .next()
        .expect("No material found in GLB");

    // baseColor
    let base_color_tex_index = material
        .pbr_metallic_roughness()
        .base_color_texture()
        .map(|info| info.texture().index());

    // metallic_roughness
    let metallic_roughness_tex_index = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .map(|info| info.texture().index());

    // normal
    let normal_tex_index = material.normal_texture().map(|info| info.texture().index());

    // Gltf format -> vk::Format
    fn gltf_format_to_vk_format(fmt: gltf::image::Format) -> ash::vk::Format {
        match fmt {
            gltf::image::Format::R8G8B8A8 => ash::vk::Format::R8G8B8A8_UNORM,
            gltf::image::Format::R8G8B8 => ash::vk::Format::R8G8B8_UNORM,
            gltf::image::Format::R16G16B16A16 => ash::vk::Format::R16G16B16A16_UNORM,
            gltf::image::Format::R32G32B32A32FLOAT => ash::vk::Format::R32G32B32A32_SFLOAT,
            _ => panic!("Unsupported GLTF image format: {:?}", fmt),
        }
    }

    fn create_vk_texture_from_gltf_image(
        state: &mut VulkanState,
        gltf_image: &GltfImageData,
        mip0_width: u32,
    ) -> Texture {
        let vk_format = gltf_format_to_vk_format(gltf_image.format);
        crate::utils::create_texture_with_mipmap(
            state,
            mip0_width,
            gltf_image.width,
            gltf_image.height,
            vk_format,
            &gltf_image.pixels,
        )
    }

    // baseColor
    let base_color = base_color_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| create_vk_texture_from_gltf_image(state, img, width))
        .expect("No baseColor texture found");

    // metallic_roughness
    let metallic_roughness = metallic_roughness_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| create_vk_texture_from_gltf_image(state, img, width));

    // normal
    let normal = normal_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| create_vk_texture_from_gltf_image(state, img, width));

    GlbTexture {
        base_color,
        metallic_roughness,
        normal,
    }
}
