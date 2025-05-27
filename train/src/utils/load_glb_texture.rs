use std::path::Path as FsPath;

use gltf::image::Data as GltfImageData;
use image::GenericImageView;

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
pub fn load_glb_texture(state: &mut VulkanState, path: impl AsRef<FsPath>) -> GlbTexture {
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

    // テクスチャ作成用ヘルパ
    fn create_vk_texture_from_gltf_image(
        state: &mut VulkanState,
        gltf_image: &GltfImageData,
    ) -> Texture {
        let img = image::load_from_memory(&gltf_image.pixels).expect("Failed to decode image");
        let rgba8 = img.to_rgba8();
        let (width, height) = img.dimensions();
        create_texture_with_mipmap(
            state,
            width,
            height,
            ash::vk::Format::R8G8B8A8_UNORM,
            &rgba8,
        )
    }

    // baseColor
    let base_color = base_color_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| create_vk_texture_from_gltf_image(state, img))
        .expect("No baseColor texture found");

    // metallic_roughness
    let metallic_roughness = metallic_roughness_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| create_vk_texture_from_gltf_image(state, img));

    // normal
    let normal = normal_tex_index
        .and_then(|idx| images.get(idx))
        .map(|img| create_vk_texture_from_gltf_image(state, img));

    GlbTexture {
        base_color,
        metallic_roughness,
        normal,
    }
}
