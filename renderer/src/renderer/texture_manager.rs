use std::path::Path;

use anyhow::Result;
use ash::vk;
use exr::prelude::*;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::renderer::vulkan_state::VulkanState;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextureIndex(u32);
impl TextureIndex {
    #[allow(dead_code)]
    pub const fn invalid() -> Self {
        Self(u32::MAX)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SamplerIndex(u32);
impl SamplerIndex {
    #[allow(dead_code)]
    pub const fn invalid() -> Self {
        Self(u32::MAX)
    }
}

pub struct Texture {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Option<Allocation>,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
    #[allow(dead_code)]
    pub format: vk::Format,
}

/// Manage bindless textures and samplers
/// Bound textures that have been read once will not be destroyed during execution.
pub struct TextureManager {
    textures: Vec<Texture>,
    current_texture_index: u32,
    texture_descriptor_set_layout: vk::DescriptorSetLayout,
    texture_descriptor_pool: vk::DescriptorPool,
    texture_descriptor_set: vk::DescriptorSet,
    samplers: Vec<vk::Sampler>,
    current_sampler_index: u32,
    sampler_descriptor_set_layout: vk::DescriptorSetLayout,
    sampler_descriptor_pool: vk::DescriptorPool,
    sampler_descriptor_set: vk::DescriptorSet,
}
impl TextureManager {
    const MAX_TEXTURES: u32 = 512;
    const MAX_SAMPLERS: u32 = 512;

    pub fn new(state: &mut VulkanState) -> Result<Self> {
        // Create texture descriptor set layout
        let texture_descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(Self::MAX_TEXTURES)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
            let mut extended_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);
            let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut extended_info);
            unsafe {
                state
                    .device
                    .create_descriptor_set_layout(&create_info, None)?
            }
        };

        // Create texture descriptor pool
        let texture_descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(Self::MAX_TEXTURES)];
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(Self::MAX_TEXTURES)
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
            unsafe { state.device.create_descriptor_pool(&create_info, None)? }
        };

        // Create texture descriptor set
        let texture_descriptor_set = {
            let mut count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&[Self::MAX_TEXTURES - 1]);
            let set_layouts = [texture_descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(texture_descriptor_pool)
                .set_layouts(&set_layouts)
                .push_next(&mut count_info);
            let descriptor_sets = unsafe { state.device.allocate_descriptor_sets(&allocate_info)? };
            descriptor_sets[0]
        };

        // Create sampler descriptor set layout
        let sampler_descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(Self::MAX_SAMPLERS)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
            let mut extended_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);
            let create_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut extended_info);
            unsafe {
                state
                    .device
                    .create_descriptor_set_layout(&create_info, None)?
            }
        };

        // Create sampler descriptor pool
        let sampler_descriptor_pool = {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(Self::MAX_SAMPLERS)];
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(Self::MAX_SAMPLERS)
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
            unsafe { state.device.create_descriptor_pool(&create_info, None)? }
        };

        // Create sampler descriptor sets
        let sampler_descriptor_set = {
            let mut count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&[Self::MAX_SAMPLERS - 1]);
            let set_layouts = [sampler_descriptor_set_layout];
            let allocate_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(sampler_descriptor_pool)
                .set_layouts(&set_layouts)
                .push_next(&mut count_info);
            let descriptor_sets = unsafe { state.device.allocate_descriptor_sets(&allocate_info)? };
            descriptor_sets[0]
        };

        Ok(Self {
            textures: Vec::new(),
            current_texture_index: 0,
            texture_descriptor_set_layout,
            texture_descriptor_pool,
            texture_descriptor_set,
            samplers: Vec::new(),
            current_sampler_index: 0,
            sampler_descriptor_set_layout,
            sampler_descriptor_pool,
            sampler_descriptor_set,
        })
    }

    pub fn descriptor_set_layout(&self) -> [vk::DescriptorSetLayout; 2] {
        [
            self.texture_descriptor_set_layout,
            self.sampler_descriptor_set_layout,
        ]
    }

    pub fn descriptor_sets(&self) -> [vk::DescriptorSet; 2] {
        [self.texture_descriptor_set, self.sampler_descriptor_set]
    }

    pub fn load_texture(
        &mut self,
        state: &mut VulkanState,
        name: &str,
        width: u32,
        height: u32,
        format: vk::Format,
        image_data: &[u8],
    ) -> Result<TextureIndex> {
        if self.current_texture_index >= Self::MAX_TEXTURES {
            return Err(anyhow::anyhow!("Max texture resources exceeded"));
        }

        // Create image
        let (image, image_view, allocation) = {
            // Create scene image
            let image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .format(format)
                .mip_levels(1)
                .array_layers(1)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .samples(vk::SampleCountFlags::TYPE_1);
            let image = unsafe { state.device.create_image(&image_create_info, None) }.unwrap();

            // Allocate memory for the image
            let image_requirements = unsafe { state.device.get_image_memory_requirements(image) };
            let image_allocation = state.allocator().allocate(&AllocationCreateDesc {
                name,
                requirements: image_requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            unsafe {
                state.device.bind_image_memory(
                    image,
                    image_allocation.memory(),
                    image_allocation.offset(),
                )?
            };

            // Create image view
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let image_view = unsafe {
                state
                    .device
                    .create_image_view(&image_view_create_info, None)?
            };

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

            // create staging buffer
            let buffer_size = std::mem::size_of_val(image_data) as u64;
            let staging_buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging_buffer = unsafe {
                state
                    .device
                    .create_buffer(&staging_buffer_create_info, None)?
            };

            // Start command buffer
            unsafe {
                state
                    .device
                    .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
            }

            // Change image layout
            let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2KHR::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2KHR::NONE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
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
                    &vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers),
                );
            }

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
            data.copy_from_slice(bytemuck::cast_slice(image_data));

            // Copy the staging buffer to the image
            let buffer_copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            unsafe {
                state.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[buffer_copy_region],
                );
            }

            // Change image layout
            let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2KHR::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2KHR::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
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
                    &vk::DependencyInfo::default().image_memory_barriers(&image_memory_barriers),
                );
            }

            // End command buffer
            unsafe {
                state.device.end_command_buffer(command_buffer)?;
            }

            // Create a fence
            let fence_create_info = vk::FenceCreateInfo::default();
            let fence = unsafe { state.device.create_fence(&fence_create_info, None).unwrap() };

            // Submit the command buffer
            let buffers_for_submission = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
            unsafe {
                state
                    .device
                    .queue_submit(state.queue, &[submit_info], fence)
                    .unwrap();
                state
                    .device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .unwrap();
            }

            // Destroy the staging buffer
            unsafe {
                state.device.destroy_buffer(staging_buffer, None);
                state
                    .allocator()
                    .free(staging_buffer_allocation)
                    .expect("Failed to free staging buffer allocation");
            }

            // Destroy the fence and command buffer
            unsafe {
                state.device.destroy_fence(fence, None);
                state
                    .device
                    .free_command_buffers(state.command_pool, &[command_buffer]);
            }

            (image, image_view, image_allocation)
        };

        // Create a texture object
        let texture = Texture {
            image,
            image_view,
            allocation: Some(allocation),
            width,
            height,
            format,
        };
        self.textures.push(texture);

        // Bind the image view to the descriptor set
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)];
        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(self.texture_descriptor_set)
            .dst_array_element(self.current_texture_index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(0)
            .image_info(&image_info);
        unsafe {
            state
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[]);
        }
        let index = TextureIndex(self.current_texture_index);
        self.current_texture_index += 1;

        Ok(index)
    }

    pub fn load_latent_texture(
        &mut self,
        state: &mut VulkanState,
        path: impl AsRef<Path>,
    ) -> Result<TextureIndex> {
        let path = path.as_ref();

        // load mip images
        let mut mip_images = vec![];
        let mut mip_level = 0;
        let mut mip0_width = 0;
        let mut mip0_height = 0;
        loop {
            let mip_path = path.with_file_name(format!(
                "{}.mip{}.exr",
                path.file_stem().unwrap().to_string_lossy(),
                mip_level
            ));
            if !mip_path.exists() {
                break;
            }
            // load as exr f16 RGBA
            let image = read_first_rgba_layer_from_file(
                &mip_path,
                |size, _| vec![vec![[half::f16::ZERO; 4]; size.x()]; size.y()],
                |pixels, position, (r, g, b, a)| {
                    pixels[position.y()][position.x()] = [r, g, b, a];
                },
            )?;
            let width = image.layer_data.size.width();
            let height = image.layer_data.size.height();
            let data = image
                .layer_data
                .channel_data
                .pixels
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            mip_images.push(data);
            if mip_level == 0 {
                mip0_width = width as u32;
                mip0_height = height as u32;
            }
            mip_level += 1;
        }
        if mip_images.is_empty() {
            anyhow::bail!("No mip images found for {:?}", path);
        }
        let mip_levels = mip_images.len() as u32;

        // Create vk::Image
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .extent(vk::Extent3D {
                width: mip0_width,
                height: mip0_height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { state.device.create_image(&image_create_info, None)? };
        let mem_req = unsafe { state.device.get_image_memory_requirements(image) };
        let allocation =
            state
                .allocator()
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "latent texture",
                    requirements: mem_req,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })?;
        unsafe {
            state
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?;
        }

        // Upload mip image data
        for (level, mip) in mip_images.iter().enumerate() {
            let command_buffer = state.begin_single_time_commands();

            let w = (mip0_width >> level).max(1);
            let h = (mip0_height >> level).max(1);
            let flat: Vec<u8> = mip
                .iter()
                .flat_map(|px| px.iter().flat_map(|c| c.to_le_bytes()))
                .collect();

            // staging buffer
            let buffer_info = vk::BufferCreateInfo::default()
                .size(flat.len() as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging_buffer = unsafe { state.device.create_buffer(&buffer_info, None)? };
            let staging_mem_req =
                unsafe { state.device.get_buffer_memory_requirements(staging_buffer) };
            let mut staging_alloc =
                state
                    .allocator()
                    .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                        name: "latent texture staging",
                        requirements: staging_mem_req,
                        location: gpu_allocator::MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme:
                            gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                    })?;
            unsafe {
                state.device.bind_buffer_memory(
                    staging_buffer,
                    staging_alloc.memory(),
                    staging_alloc.offset(),
                )?;
            }
            staging_alloc
                .mapped_slice_mut()
                .expect("Failed to map staging buffer memory")
                .copy_from_slice(&flat);

            // Barrier
            let subresource = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: level as u32,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let barrier = vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(subresource)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            unsafe {
                state.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            // Copy
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(level as u32)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            unsafe {
                state.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }

            // Change image layout to SHADER_READ_ONLY_OPTIMAL
            let barrier2 = vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(subresource)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            unsafe {
                state.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier2],
                );
            }
            state.end_single_time_commands(command_buffer);

            unsafe {
                state.device.destroy_buffer(staging_buffer, None);
            }
            state.allocator().free(staging_alloc)?;
        }

        // Create image view
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
        let image_view = unsafe { state.device.create_image_view(&image_view_info, None)? };

        // 6. Texture構造体をpush
        let texture = Texture {
            image,
            image_view,
            allocation: Some(allocation),
            width: mip0_width,
            height: mip0_height,
            format: vk::Format::R16G16B16A16_SFLOAT,
        };
        self.textures.push(texture);

        // Bind the image view to the descriptor set
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)];
        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(self.texture_descriptor_set)
            .dst_array_element(self.current_texture_index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(0)
            .image_info(&image_info);
        unsafe {
            state
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[]);
        }
        let index = TextureIndex(self.current_texture_index);
        self.current_texture_index += 1;

        Ok(index)
    }

    pub fn create_sampler(
        &mut self,
        state: &mut VulkanState,
        create_info: &vk::SamplerCreateInfo,
    ) -> Result<SamplerIndex> {
        if self.current_sampler_index >= Self::MAX_SAMPLERS {
            return Err(anyhow::anyhow!("Max sampler resources exceeded"));
        }

        // Create sampler
        let sampler = unsafe { state.device.create_sampler(create_info, None)? };

        // Bind the sampler to the descriptor set
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(self.sampler_descriptor_set)
            .dst_array_element(self.current_sampler_index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .dst_binding(0)
            .image_info(&sampler_info);
        unsafe {
            state
                .device
                .update_descriptor_sets(&[write_descriptor_set], &[]);
        }

        self.samplers.push(sampler);
        let index = SamplerIndex(self.current_sampler_index);
        self.current_sampler_index += 1;
        Ok(index)
    }

    pub fn destroy(&mut self, state: &mut VulkanState) {
        for texture in &mut self.textures {
            unsafe {
                state.device.destroy_image_view(texture.image_view, None);
                state.device.destroy_image(texture.image, None);
                let allocation = texture
                    .allocation
                    .take()
                    .expect("Failed to get texture allocation");
                state
                    .allocator()
                    .free(allocation)
                    .expect("Failed to free texture allocation");
            }
        }
        self.textures.clear();

        for sampler in &self.samplers {
            unsafe {
                state.device.destroy_sampler(*sampler, None);
            }
        }
        self.samplers.clear();

        unsafe {
            state
                .device
                .destroy_descriptor_set_layout(self.texture_descriptor_set_layout, None);
            state
                .device
                .destroy_descriptor_pool(self.texture_descriptor_pool, None);
            state
                .device
                .destroy_descriptor_set_layout(self.sampler_descriptor_set_layout, None);
            state
                .device
                .destroy_descriptor_pool(self.sampler_descriptor_pool, None);
        }
    }
}
