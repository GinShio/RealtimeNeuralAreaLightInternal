use std::path::Path;

use anyhow::Result;
use ash::vk;
use gltf::{Node, buffer};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use super::{vertex::Vertex, vulkan_state::VulkanState};

pub struct ModelData {
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub index_count: u32,
    vertex_buffer_allocation: Option<Allocation>,
    index_buffer_allocation: Option<Allocation>,
}
impl ModelData {
    pub fn new(state: &mut VulkanState, vertices: &[Vertex], indices: &[u32]) -> Result<Self> {
        // Create vertex buffer
        let (vertex_buffer, vertex_buffer_allocation) = {
            let buffer_size = std::mem::size_of_val(vertices) as u64;

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
                    name: "vertex staging buffer",
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
            data.copy_from_slice(bytemuck::cast_slice(vertices));

            // Create vertex buffer
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let vertex_buffer = unsafe { state.device.create_buffer(&buffer_create_info, None)? };

            // Allocate memory for the vertex buffer
            let vertex_buffer_requirements =
                unsafe { state.device.get_buffer_memory_requirements(vertex_buffer) };
            let vertex_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
                name: "vertex buffer",
                requirements: vertex_buffer_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            // Bind the vertex buffer memory
            unsafe {
                state.device.bind_buffer_memory(
                    vertex_buffer,
                    vertex_buffer_allocation.memory(),
                    vertex_buffer_allocation.offset(),
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
                        .dst_buffer(vertex_buffer)
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

            // Return the vertex buffer and its memory
            (vertex_buffer, vertex_buffer_allocation)
        };

        // create index buffer
        let (index_buffer, index_buffer_allocation) = {
            let buffer_size = std::mem::size_of_val(indices) as u64;

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
                    name: "vertex staging buffer",
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
            data.copy_from_slice(bytemuck::cast_slice(indices));

            // Create index buffer
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let index_buffer = unsafe { state.device.create_buffer(&buffer_create_info, None)? };

            // Allocate memory for the index buffer
            let index_buffer_requirements =
                unsafe { state.device.get_buffer_memory_requirements(index_buffer) };
            let index_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
                name: "index buffer",
                requirements: index_buffer_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            // Bind the index buffer memory
            unsafe {
                state.device.bind_buffer_memory(
                    index_buffer,
                    index_buffer_allocation.memory(),
                    index_buffer_allocation.offset(),
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
                        .dst_buffer(index_buffer)
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

            // Return the index buffer and its memory
            (index_buffer, index_buffer_allocation)
        };

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),
            index_buffer_allocation: Some(index_buffer_allocation),
        })
    }

    pub fn destroy(&mut self, state: &mut VulkanState) {
        if let Some(allocation) = self.vertex_buffer_allocation.take() {
            state.allocator().free(allocation).unwrap();
        }
        if let Some(allocation) = self.index_buffer_allocation.take() {
            state.allocator().free(allocation).unwrap();
        }
        unsafe {
            state.device.destroy_buffer(self.vertex_buffer, None);
            state.device.destroy_buffer(self.index_buffer, None);
        }
    }
}

pub fn load_glb(state: &mut VulkanState, path: impl AsRef<Path>) -> Result<Vec<ModelData>> {
    let mut model_data = vec![];

    fn traverse_gltf(
        state: &mut VulkanState,
        model_data: &mut Vec<ModelData>,
        node: Node,
        buffers: Vec<buffer::Data>,
        parent_transform: glam::Mat4,
    ) {
        let local_transform = node.transform();
        let transform =
            parent_transform * glam::Mat4::from_cols_array_2d(&local_transform.matrix());

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let indices = reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .collect::<Vec<_>>();

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
                model_data.push(ModelData::new(state, &vertices, &indices).unwrap());
            }
        }
    }

    let (document, buffers, _images) = gltf::import(path)?;
    for scene in document.scenes() {
        for node in scene.nodes() {
            traverse_gltf(
                state,
                &mut model_data,
                node,
                buffers.clone(),
                glam::Mat4::IDENTITY,
            );
        }
    }

    Ok(model_data)
}
