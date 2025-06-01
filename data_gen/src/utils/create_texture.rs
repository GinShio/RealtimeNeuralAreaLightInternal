use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use image::{DynamicImage, imageops::FilterType};

use crate::vulkan_state::VulkanState;

pub struct Texture {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Option<Allocation>,
    #[allow(dead_code)]
    pub width: u32,
}
impl Texture {
    pub fn destroy(&mut self, state: &mut VulkanState) {
        if let Some(allocation) = self.allocation.take() {
            state.allocator().free(allocation).unwrap();
        }
        unsafe {
            state.device.destroy_image_view(self.image_view, None);
            state.device.destroy_image(self.image, None);
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MipmapPushConstant {
    src_mip_level: u32,
    dst_mip_level: u32,
    src_width: u32,
    src_height: u32,
}

pub fn create_texture_with_mipmap(state: &mut VulkanState, mip0_width: u32, path: &str) -> Texture {
    // Load image data
    let img = image::open(path).expect("Failed to open image");

    // Resize to mip0_width x mip0_width
    let resized = img.resize_exact(mip0_width, mip0_width, FilterType::Lanczos3);

    // Convert resized image to byte array for Vulkan
    let resized_data = match &resized {
        DynamicImage::ImageRgba8(img) => img.as_raw().clone(),
        _ => panic!("Unsupported image format for mipmap generation"),
    };

    let width = mip0_width;
    let height = mip0_width;
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    // Create vk::Image
    let image_create_info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::empty())
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { state.device.create_image(&image_create_info, None).unwrap() };

    // Allocate memory for the image
    let mem_req = unsafe { state.device.get_image_memory_requirements(image) };
    let allocation = state
        .allocator()
        .allocate(&AllocationCreateDesc {
            name: "TextureImage",
            requirements: mem_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();
    unsafe {
        state
            .device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .unwrap();
    }

    // Create image view
    let image_view = {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            state
                .device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        }
    };

    // Prepare staging buffer using final_data
    let buffer_size = resized_data.len() as vk::DeviceSize;
    let staging_buffer_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging_buffer = unsafe {
        state
            .device
            .create_buffer(&staging_buffer_info, None)
            .unwrap()
    };
    let staging_mem_req = unsafe { state.device.get_buffer_memory_requirements(staging_buffer) };
    let mut staging_allocation = state
        .allocator()
        .allocate(&AllocationCreateDesc {
            name: "StagingBuffer",
            requirements: staging_mem_req,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();
    unsafe {
        state
            .device
            .bind_buffer_memory(
                staging_buffer,
                staging_allocation.memory(),
                staging_allocation.offset(),
            )
            .unwrap();
    }

    // Write resized_data to staging buffer
    staging_allocation
        .mapped_slice_mut()
        .expect("Failed to map staging buffer memory")
        .copy_from_slice(&resized_data);

    // Copy staging buffer to image mip0
    let cmd = state.begin_single_time_commands();

    // Transition image layout: UNDEFINED -> TRANSFER_DST_OPTIMAL
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        state.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // Buffer image copy
    let buffer_image_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });
    unsafe {
        state.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[buffer_image_copy],
        );
    }

    // Transition image layout: TRANSFER_DST_OPTIMAL -> GENERAL (for compute shader)
    let barrier2 = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        state.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier2],
        );
    }
    state.end_single_time_commands(cmd);

    // Clean up staging buffer
    state.allocator().free(staging_allocation).unwrap();
    unsafe {
        state.device.destroy_buffer(staging_buffer, None);
    }

    // Create image views for each mip level for mipmap generation
    let mut mip_image_views = Vec::with_capacity(mip_levels as usize);
    for mip in 0..mip_levels {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let image_view = unsafe {
            state
                .device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };
        mip_image_views.push(image_view);
    }

    // Image layout transition: UNDEFINED -> GENERAL for all mip levels
    {
        let cmd = state.begin_single_time_commands();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            state.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        state.end_single_time_commands(cmd);
    }

    // Descriptor set layout for 2 storage images (input/output)
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe {
        state
            .device
            .create_descriptor_set_layout(&set_layout_info, None)
            .unwrap()
    };

    // Descriptor pool
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(2)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
    let descriptor_pool = unsafe {
        state
            .device
            .create_descriptor_pool(&pool_info, None)
            .unwrap()
    };

    // Descriptor set
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&set_layout));
    let descriptor_set = unsafe { state.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

    // Create Compute Pipeline
    let push_constant_ranges = &[vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<MipmapPushConstant>() as u32)];
    let (pipeline, pipeline_layout) = crate::utils::create_pipeline::create_compute_pipeline(
        state,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/utils/generate_mipmap.comp.spv"
        )),
        &[set_layout],
        push_constant_ranges,
    )
    .unwrap();

    // Mipmap generation loop
    let mut src_width = width;
    let mut src_height = height;
    for mip in 1..mip_levels {
        // Update descriptor set for src/dst image views
        let input_image_info = vk::DescriptorImageInfo::default()
            .image_view(mip_image_views[(mip - 1) as usize])
            .image_layout(vk::ImageLayout::GENERAL);
        let output_image_info = vk::DescriptorImageInfo::default()
            .image_view(mip_image_views[mip as usize])
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_image_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_image_info)),
        ];
        unsafe {
            state.device.update_descriptor_sets(&writes, &[]);
        }

        // Push constant
        let push_constant = MipmapPushConstant {
            src_mip_level: mip - 1,
            dst_mip_level: mip,
            src_width,
            src_height,
        };

        // Command buffer for this mip
        let cmd = state.begin_single_time_commands();
        unsafe {
            state
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            state.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            state.device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constant),
            );
            let group_count_x = (src_width / 2).max(1);
            let group_count_y = (src_height / 2).max(1);
            state
                .device
                .cmd_dispatch(cmd, group_count_x, group_count_y, 1);
        }
        state.end_single_time_commands(cmd);

        src_width = (src_width / 2).max(1);
        src_height = (src_height / 2).max(1);
    }

    // Transition image layout for shader read
    {
        let cmd = state.begin_single_time_commands();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            state.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        state.end_single_time_commands(cmd);
    }

    // Cleanup
    unsafe {
        state.device.destroy_pipeline(pipeline, None);
        state.device.destroy_pipeline_layout(pipeline_layout, None);
        state.device.destroy_descriptor_pool(descriptor_pool, None);
        state.device.destroy_descriptor_set_layout(set_layout, None);
        for view in mip_image_views {
            state.device.destroy_image_view(view, None);
        }
    }

    Texture {
        image,
        image_view,
        allocation: Some(allocation),
        width,
    }
}

pub fn create_texture_with_mipmap_data(
    state: &mut VulkanState,
    mip0_width: u32,
    width: u32,
    height: u32,
    format: vk::Format,
    data: &[u8],
) -> Texture {
    // Create DynamicImage from vk::Format
    let img = match format {
        vk::Format::R8G8B8A8_UNORM => image::RgbaImage::from_raw(width, height, data.to_vec())
            .map(DynamicImage::ImageRgba8)
            .expect("Invalid RGBA8 image"),
        _ => image::load_from_memory(data).expect("Failed to decode image"),
    };

    // Resize to mip0_width x mip0_width
    let resized = img.resize_exact(mip0_width, mip0_width, FilterType::Lanczos3);

    // Convert resized image to byte array for Vulkan
    let resized_data = match &resized {
        DynamicImage::ImageRgba8(img) => img.as_raw().clone(),
        _ => panic!("Unsupported image format for mipmap generation"),
    };

    let width = mip0_width;
    let height = mip0_width;
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    // Create vk::Image
    let image_create_info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::empty())
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { state.device.create_image(&image_create_info, None).unwrap() };

    // Allocate memory for the image
    let mem_req = unsafe { state.device.get_image_memory_requirements(image) };
    let allocation = state
        .allocator()
        .allocate(&AllocationCreateDesc {
            name: "TextureImage",
            requirements: mem_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();
    unsafe {
        state
            .device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .unwrap();
    }

    // Create image view
    let image_view = {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            state
                .device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        }
    };

    // Prepare staging buffer using final_data
    let buffer_size = resized_data.len() as vk::DeviceSize;
    let staging_buffer_info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::empty())
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging_buffer = unsafe {
        state
            .device
            .create_buffer(&staging_buffer_info, None)
            .unwrap()
    };
    let staging_mem_req = unsafe { state.device.get_buffer_memory_requirements(staging_buffer) };
    let mut staging_allocation = state
        .allocator()
        .allocate(&AllocationCreateDesc {
            name: "StagingBuffer",
            requirements: staging_mem_req,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .unwrap();
    unsafe {
        state
            .device
            .bind_buffer_memory(
                staging_buffer,
                staging_allocation.memory(),
                staging_allocation.offset(),
            )
            .unwrap();
    }

    // Write resized_data to staging buffer
    staging_allocation
        .mapped_slice_mut()
        .expect("Failed to map staging buffer memory")
        .copy_from_slice(&resized_data);

    // Copy staging buffer to image mip0
    let cmd = state.begin_single_time_commands();

    // Transition image layout: UNDEFINED -> TRANSFER_DST_OPTIMAL
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        state.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // Buffer image copy
    let buffer_image_copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });
    unsafe {
        state.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[buffer_image_copy],
        );
    }

    // Transition image layout: TRANSFER_DST_OPTIMAL -> GENERAL (for compute shader)
    let barrier2 = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        state.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier2],
        );
    }
    state.end_single_time_commands(cmd);

    // Clean up staging buffer
    state.allocator().free(staging_allocation).unwrap();
    unsafe {
        state.device.destroy_buffer(staging_buffer, None);
    }

    // Create image views for each mip level for mipmap generation
    let mut mip_image_views = Vec::with_capacity(mip_levels as usize);
    for mip in 0..mip_levels {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let image_view = unsafe {
            state
                .device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        };
        mip_image_views.push(image_view);
    }

    // Image layout transition: UNDEFINED -> GENERAL for all mip levels
    {
        let cmd = state.begin_single_time_commands();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            state.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        state.end_single_time_commands(cmd);
    }

    // Descriptor set layout for 2 storage images (input/output)
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe {
        state
            .device
            .create_descriptor_set_layout(&set_layout_info, None)
            .unwrap()
    };

    // Descriptor pool
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(2)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
    let descriptor_pool = unsafe {
        state
            .device
            .create_descriptor_pool(&pool_info, None)
            .unwrap()
    };

    // Descriptor set
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&set_layout));
    let descriptor_set = unsafe { state.device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

    // Create Compute Pipeline
    let push_constant_ranges = &[vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<MipmapPushConstant>() as u32)];
    let (pipeline, pipeline_layout) = crate::utils::create_pipeline::create_compute_pipeline(
        state,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/utils/generate_mipmap.comp.spv"
        )),
        &[set_layout],
        push_constant_ranges,
    )
    .unwrap();

    // Mipmap generation loop
    let mut src_width = width;
    let mut src_height = height;
    for mip in 1..mip_levels {
        // Update descriptor set for src/dst image views
        let input_image_info = vk::DescriptorImageInfo::default()
            .image_view(mip_image_views[(mip - 1) as usize])
            .image_layout(vk::ImageLayout::GENERAL);
        let output_image_info = vk::DescriptorImageInfo::default()
            .image_view(mip_image_views[mip as usize])
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&input_image_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_image_info)),
        ];
        unsafe {
            state.device.update_descriptor_sets(&writes, &[]);
        }

        // Push constant
        let push_constant = MipmapPushConstant {
            src_mip_level: mip - 1,
            dst_mip_level: mip,
            src_width,
            src_height,
        };

        // Command buffer for this mip
        let cmd = state.begin_single_time_commands();
        unsafe {
            state
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            state.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            state.device.cmd_push_constants(
                cmd,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constant),
            );
            let group_count_x = (src_width / 2).max(1);
            let group_count_y = (src_height / 2).max(1);
            state
                .device
                .cmd_dispatch(cmd, group_count_x, group_count_y, 1);
        }
        state.end_single_time_commands(cmd);

        src_width = (src_width / 2).max(1);
        src_height = (src_height / 2).max(1);
    }

    // Transition image layout for shader read
    {
        let cmd = state.begin_single_time_commands();
        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            state.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        state.end_single_time_commands(cmd);
    }

    // Cleanup
    unsafe {
        state.device.destroy_pipeline(pipeline, None);
        state.device.destroy_pipeline_layout(pipeline_layout, None);
        state.device.destroy_descriptor_pool(descriptor_pool, None);
        state.device.destroy_descriptor_set_layout(set_layout, None);
        for view in mip_image_views {
            state.device.destroy_image_view(view, None);
        }
    }

    Texture {
        image,
        image_view,
        allocation: Some(allocation),
        width,
    }
}
