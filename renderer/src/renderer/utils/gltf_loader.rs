use std::path::Path;

use anyhow::Result;
use ash::vk;
use gltf::{Node, buffer};

use crate::renderer::{
    model_data::ModelData,
    texture_manager::{TextureIndex, TextureManager},
    vertex::Vertex,
    vulkan_state::VulkanState,
};

pub struct GltfTextures {
    pub base_color: Option<TextureIndex>,
    pub normal: Option<TextureIndex>,
    pub metallic_roughness: Option<TextureIndex>,
}

pub fn load_glb(
    state: &mut VulkanState,
    texture_manager: &mut TextureManager,
    path: impl AsRef<Path>,
) -> Result<Vec<(ModelData, GltfTextures)>> {
    let mut model_data = vec![];

    fn traverse_gltf(
        state: &mut VulkanState,
        texture_manager: &mut TextureManager,
        model_data: &mut Vec<(ModelData, GltfTextures)>,
        node: Node,
        buffers: &[buffer::Data],
        images: &[gltf::image::Data],
        parent_transform: glam::Mat4,
    ) {
        let local_transform = node.transform();
        let transform =
            parent_transform * glam::Mat4::from_cols_array_2d(&local_transform.matrix());

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                // get name
                let name = node.name().unwrap_or("Unnamed").to_string();

                // get glb buffer data
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // get indices
                let indices = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .collect::<Vec<_>>();

                // get vertices
                let mut vertices = vec![];

                let positions = reader
                    .read_positions()
                    .unwrap()
                    .map(glam::Vec3::from)
                    .map(|p| transform.transform_point3(p))
                    .collect::<Vec<_>>();
                let normals = reader
                    .read_normals()
                    .unwrap()
                    .map(glam::Vec3::from)
                    .map(|n| {
                        glam::Mat3::from_mat4(transform)
                            .inverse()
                            .transpose()
                            .mul_vec3(n)
                    })
                    .collect::<Vec<_>>();
                let uvs = if let Some(uvs) = reader.read_tex_coords(0) {
                    uvs.into_f32().map(glam::Vec2::from).collect::<Vec<_>>()
                } else {
                    vec![glam::Vec2::ZERO; positions.len()]
                };

                let is_mirrored = transform.determinant() < 0.0;
                let tangents = if let Some(tangents) = reader.read_tangents() {
                    tangents
                        .map(glam::Vec4::from)
                        .map(|t| {
                            let tt = glam::Mat3::from_mat4(transform)
                                .inverse()
                                .transpose()
                                .mul_vec3(t.truncate());
                            let w = if is_mirrored { -t.w } else { t.w };
                            glam::Vec4::new(tt.x, tt.y, tt.z, w)
                        })
                        .collect::<Vec<_>>()
                } else {
                    let mut tangents = vec![glam::Vec4::ZERO; positions.len()];
                    for is in indices.chunks(3) {
                        let i0 = is[0] as usize;
                        let i1 = is[1] as usize;
                        let i2 = is[2] as usize;

                        let p0 = positions[i0];
                        let p1 = positions[i1];
                        let p2 = positions[i2];

                        let uv0 = uvs[i0];
                        let uv1 = uvs[i1];
                        let uv2 = uvs[i2];

                        let edge1 = p1 - p0;
                        let edge2 = p2 - p0;

                        let delta_uv1 = uv1 - uv0;
                        let delta_uv2 = uv2 - uv0;

                        let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);

                        let normal = edge1.cross(edge2).normalize();
                        let tangent = ((edge1 * delta_uv2.y - edge2 * delta_uv1.y) * r).normalize();
                        let bitangnet =
                            ((edge2 * delta_uv1.x - edge1 * delta_uv2.x) * r).normalize();

                        let w = if normal.cross(tangent).dot(bitangnet) < 0.0 {
                            -1.0
                        } else {
                            1.0
                        };

                        let tangent = glam::Vec4::new(tangent.x, tangent.y, tangent.z, w);

                        tangents[i0] = tangent;
                        tangents[i1] = tangent;
                        tangents[i2] = tangent;
                    }
                    tangents
                };

                for i in 0..positions.len() {
                    vertices.push(Vertex {
                        pos: positions[i].into(),
                        normal: normals[i].into(),
                        tangent: tangents[i].into(),
                        uv: uvs[i].into(),
                    });
                }

                // create model data
                let model = ModelData::new(state, &vertices, &indices).unwrap();

                // get textures
                let pbr_material = primitive.material().pbr_metallic_roughness();

                let base_color = pbr_material
                    .base_color_texture()
                    .and_then(|info| {
                        let tex_index = info.texture().index();
                        images.get(tex_index)
                    })
                    .map(|image_data| {
                        let width = image_data.width;
                        let height = image_data.height;
                        let format = vk::Format::R8G8B8A8_SRGB;
                        let mut data = Vec::with_capacity((width * height * 4) as usize);
                        match image_data.format {
                            gltf::image::Format::R8G8B8A8 => {
                                data.extend_from_slice(&image_data.pixels)
                            }
                            gltf::image::Format::R8G8B8 => {
                                data.extend(image_data.pixels.chunks(3).flat_map(|p| {
                                    let r = p[0];
                                    let g = p[1];
                                    let b = p[2];
                                    vec![r, g, b, 255]
                                }))
                            }
                            _ => panic!("Unsupported image format"),
                        };
                        texture_manager
                            .load_texture(
                                state,
                                &format!("{}.base_color", name),
                                width,
                                height,
                                format,
                                &data,
                            )
                            .expect("Failed to load texture")
                    });

                let normal = primitive
                    .material()
                    .normal_texture()
                    .and_then(|info| {
                        let tex_index = info.texture().index();
                        images.get(tex_index)
                    })
                    .map(|image_data| {
                        let width = image_data.width;
                        let height = image_data.height;
                        let format = vk::Format::R8G8B8A8_UNORM;
                        let mut data = Vec::with_capacity((width * height * 4) as usize);
                        match image_data.format {
                            gltf::image::Format::R8G8B8A8 => {
                                data.extend_from_slice(&image_data.pixels)
                            }
                            gltf::image::Format::R8G8B8 => {
                                data.extend(image_data.pixels.chunks(3).flat_map(|p| {
                                    let r = p[0];
                                    let g = p[1];
                                    let b = p[2];
                                    vec![r, g, b, 255]
                                }))
                            }
                            _ => panic!("Unsupported image format"),
                        };
                        texture_manager
                            .load_texture(
                                state,
                                &format!("{}.normal", name),
                                width,
                                height,
                                format,
                                &data,
                            )
                            .expect("Failed to load texture")
                    });

                let metallic_roughness = pbr_material
                    .metallic_roughness_texture()
                    .and_then(|info| {
                        let tex_index = info.texture().index();
                        images.get(tex_index)
                    })
                    .map(|image_data| {
                        let width = image_data.width;
                        let height = image_data.height;
                        let format = vk::Format::R8G8B8A8_UNORM;
                        let mut data = Vec::with_capacity((width * height * 4) as usize);
                        match image_data.format {
                            gltf::image::Format::R8G8B8A8 => {
                                data.extend_from_slice(&image_data.pixels)
                            }
                            gltf::image::Format::R8G8B8 => {
                                data.extend(image_data.pixels.chunks(3).flat_map(|p| {
                                    let r = p[0];
                                    let g = p[1];
                                    let b = p[2];
                                    vec![r, g, b, 255]
                                }))
                            }
                            _ => panic!("Unsupported image format"),
                        };
                        texture_manager
                            .load_texture(
                                state,
                                &format!("{}.metallic_roughness", name),
                                width,
                                height,
                                format,
                                &data,
                            )
                            .expect("Failed to load texture")
                    });

                let textures = GltfTextures {
                    base_color,
                    normal,
                    metallic_roughness,
                };

                // push model data
                model_data.push((model, textures));
            }
        }
    }

    let (document, buffers, images) = gltf::import(path)?;
    for scene in document.scenes() {
        for node in scene.nodes() {
            traverse_gltf(
                state,
                texture_manager,
                &mut model_data,
                node,
                &buffers,
                &images,
                glam::Mat4::IDENTITY,
            );
        }
    }

    Ok(model_data)
}
