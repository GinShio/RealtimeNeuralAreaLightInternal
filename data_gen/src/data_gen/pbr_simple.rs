use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use ash::vk;
use rand::prelude::*;

use crate::{
    utils::{
        create_compute_pipeline, create_cpu_storage_buffer, create_storage_buffer,
        create_texture_with_mipmap, create_uniform_buffer,
    },
    vulkan_state::VulkanState,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBuffer {
    data_size: u32,
    texture_size: u32,
    max_light_size: f32,
    max_light_distance: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FirstPhasePushConstants {
    seed: u64,
    mollification_scale: f32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SecondPhasePushConstants {
    seed: u64,
    _padding: [u32; 2],
}

pub fn data_gen(
    state: &mut VulkanState,
    base_color_texture_path: &str,
    metallic_texture_path: &str,
    roughness_texture_path: &str,
    normal_texture_path: &str,
    texture_size: u32,
    batch_size: u64,
    first_phase_shard_size: u64,
    mollification_shard_count: u64,
    first_phase_shard_count: u64,
    second_phase_shard_size: u64,
    second_phase_shard_count: u64,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
    }

    // Output json file
    let output_json_path = output_dir.join("data_gen_config.json");
    let config = serde_json::json!({
        "texture_size": texture_size,
        "batch_size": batch_size,
        "first_phase_shard_size": first_phase_shard_size,
        "mollification_shard_count": mollification_shard_count,
        "first_phase_shard_count": first_phase_shard_count,
        "second_phase_shard_size": second_phase_shard_size,
        "second_phase_shard_count": second_phase_shard_count,
    });
    let mut file = File::create(output_json_path)?;
    file.write_all(config.to_string().as_bytes())
        .expect("Failed to write config JSON");

    let max_light_size = 3.0;
    let max_light_distance = 10.0;

    // base_color (3)
    // roughness (1)
    // metallic (1)
    // normal (3)
    // wo (3)
    // vertex direction + distance ((3 + 1) * 4)
    // area (1)
    // Distribution (3)
    let first_shard_data_component_size = 31;

    // base_color (3)
    // roughness (1)
    // metallic (1)
    // normal (3)
    let second_material_data_component_size = 8;

    // wo (3)
    // vertex direction + distance ((3 + 1) * 4)
    // area (1)
    // Distribution (3)
    let second_shard_data_component_size = 23;

    let texture_total_pixel_size = {
        let mut pixel_count = 0;
        let mut width = texture_size as u64;
        while width > 0 {
            pixel_count += width * width;
            width /= 2;
        }
        pixel_count
    };

    let first_shard_buffer_size =
        batch_size * first_phase_shard_size * first_shard_data_component_size;
    let second_material_buffer_size =
        texture_total_pixel_size * second_material_data_component_size;
    let second_shard_buffer_size =
        texture_total_pixel_size * second_phase_shard_size * second_shard_data_component_size;

    // === Load textures ===
    let mut base_color_texture =
        create_texture_with_mipmap(state, texture_size, base_color_texture_path);
    let mut metallic_texture =
        create_texture_with_mipmap(state, texture_size, metallic_texture_path);
    let mut roughness_texture =
        create_texture_with_mipmap(state, texture_size, roughness_texture_path);
    let mut normal_texture = create_texture_with_mipmap(state, texture_size, normal_texture_path);

    // === Initialize sampler, buffers and pipelines ===

    let sampler = {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT);
        unsafe {
            state
                .device
                .create_sampler(&info, None)
                .expect("Create sampler failed")
        }
    };

    // Prepare push constants ranges
    let first_phase_push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<FirstPhasePushConstants>() as u32)];
    let second_phase_push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<SecondPhasePushConstants>() as u32)];

    // Create uniform buffers
    let (uniform_buffer, mut uniform_buffer_allocation) =
        create_uniform_buffer(state, std::mem::size_of::<UniformBuffer>() as u64)?;

    // Create storage buffers
    let (first_phase_data_buffer, first_phase_data_buffer_allocation) = create_storage_buffer(
        state,
        first_shard_buffer_size * std::mem::size_of::<f32>() as u64,
    )?;
    let (first_phase_data_cpu_buffer, first_phase_data_cpu_buffer_allocation) =
        create_cpu_storage_buffer(
            state,
            first_shard_buffer_size * std::mem::size_of::<f32>() as u64,
        )?;

    let (second_phase_material_data_buffer, second_phase_material_data_buffer_allocation) =
        create_storage_buffer(
            state,
            second_material_buffer_size * std::mem::size_of::<f32>() as u64,
        )?;
    let (second_phase_material_data_cpu_buffer, second_phase_material_data_cpu_buffer_allocation) =
        create_cpu_storage_buffer(
            state,
            second_material_buffer_size * std::mem::size_of::<f32>() as u64,
        )?;

    let (second_phase_data_buffer, second_phase_data_buffer_allocation) = create_storage_buffer(
        state,
        second_shard_buffer_size * std::mem::size_of::<f32>() as u64,
    )?;
    let (second_phase_data_cpu_buffer, second_phase_data_cpu_buffer_allocation) =
        create_cpu_storage_buffer(
            state,
            second_shard_buffer_size * std::mem::size_of::<f32>() as u64,
        )?;

    // Create descriptor layouts
    let first_phase_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // data
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // base color texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // roughness metallic texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // metallic metallic texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // normal texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            state
                .device
                .create_descriptor_set_layout(&create_info, None)
                .expect("Create descriptor set layout failed")
        }
    };
    let second_phase_material_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // data
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // base color texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // roughness metallic texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // metallic metallic texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // normal texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            state
                .device
                .create_descriptor_set_layout(&create_info, None)
                .expect("Create descriptor set layout failed")
        }
    };
    let second_phase_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // data
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // material data
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        unsafe {
            state
                .device
                .create_descriptor_set_layout(&create_info, None)
                .expect("Create descriptor set layout failed")
        }
    };

    // Create descriptor pools
    let first_phase_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(4),
        ];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_size)
            .max_sets(1);
        unsafe {
            state
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Create descriptor pool failed")
        }
    };
    let second_phase_material_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(4),
        ];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_size)
            .max_sets(1);
        unsafe {
            state
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Create descriptor pool failed")
        }
    };
    let second_phase_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
        ];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_size)
            .max_sets(1);
        unsafe {
            state
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Create descriptor pool failed")
        }
    };

    // Create descriptor set
    let first_phase_descriptor_set = {
        let set_layouts = [first_phase_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(first_phase_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    let second_phase_material_descriptor_set = {
        let set_layouts = [second_phase_material_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(second_phase_material_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    let second_phase_descriptor_set = {
        let set_layouts = [second_phase_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(second_phase_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];

    // Update descriptor sets
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBuffer>() as u64)];

        let first_data_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(first_phase_data_buffer)
            .offset(0)
            .range(first_shard_buffer_size * std::mem::size_of::<f32>() as u64)];

        let base_color_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(base_color_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let roughness_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(roughness_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let metallic_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(metallic_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let normal_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(normal_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // data buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&first_data_buffer_info),
            // base color texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&base_color_texture_info),
            // roughness texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&roughness_texture_info),
            // metallic texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&metallic_texture_info),
            // normal texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_texture_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBuffer>() as u64)];

        let second_material_data_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(second_phase_material_data_buffer)
            .offset(0)
            .range(second_material_buffer_size * std::mem::size_of::<f32>() as u64)];

        let base_color_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(base_color_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let roughness_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(roughness_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let metallic_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(metallic_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let normal_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(normal_texture.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_material_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // data buffer
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_material_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&second_material_data_buffer_info),
            // base color texture
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_material_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&base_color_texture_info),
            // roughness texture
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_material_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&roughness_texture_info),
            // metallic texture
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_material_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&metallic_texture_info),
            // normal texture
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_material_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_texture_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBuffer>() as u64)];

        let second_data_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(second_phase_data_buffer)
            .offset(0)
            .range(second_shard_buffer_size * std::mem::size_of::<f32>() as u64)];
        let material_data_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(second_phase_material_data_buffer)
            .offset(0)
            .range(second_material_buffer_size * std::mem::size_of::<f32>() as u64)];

        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // data buffer
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&second_data_buffer_info),
            // material data buffer
            vk::WriteDescriptorSet::default()
                .dst_set(second_phase_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&material_data_buffer_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    // create pipelines
    let (first_phase_pipeline, first_phase_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/pbr-simple/data-gen-1st.comp.spv"
            )),
            &[first_phase_descriptor_set_layout],
            &first_phase_push_constant_ranges,
        )?
    };
    let (second_phase_material_pipeline, second_phase_material_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/pbr-simple/data-gen-2nd-material.comp.spv"
            )),
            &[second_phase_material_descriptor_set_layout],
            &[],
        )?
    };
    let (second_phase_pipeline, second_phase_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/pbr-simple/data-gen-2nd.comp.spv"
            )),
            &[second_phase_descriptor_set_layout],
            &second_phase_push_constant_ranges,
        )?
    };

    // === Generate data ===

    let start = std::time::Instant::now();
    println!("Generating data...");
    std::io::stdout().flush().expect("Failed to flush stdout");

    let mut rng = rand::rng();

    // Generate first phase data
    println!(
        "  First phase data generation: {} shards, {} floats per shard",
        first_phase_shard_count, first_shard_buffer_size
    );
    std::io::stdout().flush().expect("Failed to flush stdout");
    let first_start = std::time::Instant::now();

    let uniform_data = UniformBuffer {
        data_size: (batch_size * first_phase_shard_size) as u32,
        texture_size: texture_size,
        max_light_size,
        max_light_distance,
    };
    uniform_buffer_allocation
        .mapped_slice_mut()
        .expect("Failed to map uniform buffer")[0..std::mem::size_of::<UniformBuffer>()]
        .copy_from_slice(bytemuck::bytes_of(&uniform_data));

    for i in 0..(mollification_shard_count + first_phase_shard_count) {
        let step_start = std::time::Instant::now();

        let seed: u64 = rng.random();

        let mollification_scale = if i < mollification_shard_count {
            i as f32 / mollification_shard_count as f32
        } else {
            0.0
        };
        let push_constants = FirstPhasePushConstants {
            seed,
            mollification_scale,
            _padding: 0,
        };

        unsafe {
            let command_buffer = state.begin_single_time_commands();

            state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                first_phase_pipeline,
            );
            state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                first_phase_pipeline_layout,
                0,
                &[first_phase_descriptor_set],
                &[],
            );
            state.device.cmd_push_constants(
                command_buffer,
                first_phase_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            state.device.cmd_dispatch(
                command_buffer,
                (batch_size * first_phase_shard_size).div_ceil(64) as u32,
                1,
                1,
            );

            // Copy data to CPU buffer

            // Barrier to ensure all writes are visible before copying
            let buffer_memory_barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .buffer(first_phase_data_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            let dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);

            state
                .device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info);

            state.device.cmd_copy_buffer(
                command_buffer,
                first_phase_data_buffer,
                first_phase_data_cpu_buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(first_phase_shard_size * std::mem::size_of::<f32>() as u64)],
            );

            state.end_single_time_commands(command_buffer);
        }

        // save shard data
        {
            let data_slice = first_phase_data_cpu_buffer_allocation
                .mapped_slice()
                .expect("Failed to map first phase data buffer");
            let mut file = File::create(output_dir.join(format!(
                "first_phase_data{}.shard-{}.bin",
                if i < mollification_shard_count {
                    "-mollified"
                } else {
                    ""
                },
                if i < mollification_shard_count {
                    i
                } else {
                    i - mollification_shard_count
                }
            )))?;
            file.write_all(data_slice)
                .expect("Failed to write first phase data shard");
        }

        let elapsed = step_start.elapsed();
        let minutes = elapsed.as_secs() / 60;
        let seconds = elapsed.as_secs() % 60;
        let millis = elapsed.subsec_millis();
        print!(
            "\r    Shard {} processed ({:02}m {:02}s {:02}ms)",
            i, minutes, seconds, millis
        );
        std::io::stdout().flush().expect("Failed to flush stdout");
    }

    let elapsed = first_start.elapsed();
    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;
    let millis = elapsed.subsec_millis();
    println!(
        "\r  First phase data generation completed ({:02}m {:02}s {:02}ms)",
        minutes, seconds, millis
    );
    std::io::stdout().flush().expect("Failed to flush stdout");

    // Generate second phase material data
    println!(
        "  Second phase material data generation: {} shards, {} floats per shard",
        second_phase_shard_count, second_material_buffer_size
    );
    std::io::stdout().flush().expect("Failed to flush stdout");
    let second_material_start = std::time::Instant::now();

    let uniform_data = UniformBuffer {
        data_size: texture_total_pixel_size as u32,
        texture_size: texture_size,
        max_light_size,
        max_light_distance,
    };
    uniform_buffer_allocation
        .mapped_slice_mut()
        .expect("Failed to map uniform buffer")[0..std::mem::size_of::<UniformBuffer>()]
        .copy_from_slice(bytemuck::bytes_of(&uniform_data));

    let command_buffer = state.begin_single_time_commands();
    unsafe {
        state.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            second_phase_material_pipeline,
        );
        state.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            second_phase_material_pipeline_layout,
            0,
            &[second_phase_material_descriptor_set],
            &[],
        );
        state.device.cmd_dispatch(
            command_buffer,
            texture_total_pixel_size.div_ceil(64) as u32,
            1,
            1,
        );

        // Copy data to CPU buffer
        let buffer_memory_barriers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .buffer(second_phase_material_data_buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)];
        let dependency_info =
            vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);

        state
            .device
            .cmd_pipeline_barrier2(command_buffer, &dependency_info);

        state.device.cmd_copy_buffer(
            command_buffer,
            second_phase_material_data_buffer,
            second_phase_material_data_cpu_buffer,
            &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(second_phase_shard_size * std::mem::size_of::<f32>() as u64)],
        );
    }
    state.end_single_time_commands(command_buffer);

    // save material data
    {
        let data_slice = first_phase_data_cpu_buffer_allocation
            .mapped_slice()
            .expect("Failed to map second phase material data buffer");
        let mut file = File::create(output_dir.join("second_phase_data.material.bin"))?;
        file.write_all(data_slice)
            .expect("Failed to write first phase material data shard");
    }

    let elapsed = second_material_start.elapsed();
    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;
    let millis = elapsed.subsec_millis();
    print!(
        "  Second phase material data generation completed ({:02}m {:02}s {:02}ms)",
        minutes, seconds, millis
    );
    std::io::stdout().flush().expect("Failed to flush stdout");

    // Generate second phase data
    println!(
        "  Second phase data generation: {} shards, {} floats per shard",
        second_phase_shard_count, second_shard_buffer_size
    );
    std::io::stdout().flush().expect("Failed to flush stdout");
    let second_start = std::time::Instant::now();

    let uniform_data = UniformBuffer {
        data_size: (texture_total_pixel_size * second_phase_shard_size) as u32,
        texture_size: texture_size,
        max_light_size,
        max_light_distance,
    };
    uniform_buffer_allocation
        .mapped_slice_mut()
        .expect("Failed to map uniform buffer")[0..std::mem::size_of::<UniformBuffer>()]
        .copy_from_slice(bytemuck::bytes_of(&uniform_data));

    for i in 0..second_phase_shard_count {
        let step_start = std::time::Instant::now();

        let seed: u64 = rng.random();

        let push_constants = SecondPhasePushConstants {
            seed,
            _padding: [0; 2],
        };

        unsafe {
            let command_buffer = state.begin_single_time_commands();

            state.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                second_phase_pipeline,
            );
            state.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                second_phase_pipeline_layout,
                0,
                &[second_phase_descriptor_set],
                &[],
            );
            state.device.cmd_push_constants(
                command_buffer,
                second_phase_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            state.device.cmd_dispatch(
                command_buffer,
                (texture_total_pixel_size * second_phase_shard_size).div_ceil(64) as u32,
                1,
                1,
            );

            // Copy data to CPU buffer
            let buffer_memory_barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .buffer(second_phase_data_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            let dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);

            state
                .device
                .cmd_pipeline_barrier2(command_buffer, &dependency_info);

            state.device.cmd_copy_buffer(
                command_buffer,
                second_phase_data_buffer,
                second_phase_data_cpu_buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(second_shard_buffer_size * std::mem::size_of::<f32>() as u64)],
            );

            state.end_single_time_commands(command_buffer);
        }

        // save shard data
        {
            let data_slice = first_phase_data_cpu_buffer_allocation
                .mapped_slice()
                .expect("Failed to map second phase data buffer");
            let mut file =
                File::create(output_dir.join(format!("second_phase_data.shard-{}.bin", i)))?;
            file.write_all(data_slice)
                .expect("Failed to write first phase data shard");
        }

        let elapsed = step_start.elapsed();
        let minutes = elapsed.as_secs() / 60;
        let seconds = elapsed.as_secs() % 60;
        let millis = elapsed.subsec_millis();
        print!(
            "\r    Shard {} processed ({:02}m {:02}s {:02}ms)",
            i, minutes, seconds, millis
        );
        std::io::stdout().flush().expect("Failed to flush stdout");
    }

    let elapsed = second_start.elapsed();
    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;
    let millis = elapsed.subsec_millis();
    println!(
        "\r  Second phase data generation completed ({:02}m {:02}s {:02}ms)",
        minutes, seconds, millis
    );
    std::io::stdout().flush().expect("Failed to flush stdout");

    let elapsed = start.elapsed();
    let minutes = elapsed.as_secs() / 60;
    let seconds = elapsed.as_secs() % 60;
    let millis = elapsed.subsec_millis();
    println!(
        "  All data generation completed! ({:02}m {:02}s {:02}ms)",
        minutes, seconds, millis
    );
    std::io::stdout().flush().expect("Failed to flush stdout");

    // === Cleanup ===

    unsafe {
        state.device.destroy_sampler(sampler, None);
        state
            .device
            .destroy_descriptor_set_layout(first_phase_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_set_layout(second_phase_material_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_set_layout(second_phase_descriptor_set_layout, None);

        state
            .device
            .destroy_descriptor_pool(first_phase_descriptor_pool, None);
        state
            .device
            .destroy_descriptor_pool(second_phase_material_descriptor_pool, None);
        state
            .device
            .destroy_descriptor_pool(second_phase_descriptor_pool, None);

        state.device.destroy_buffer(uniform_buffer, None);
        state.allocator().free(uniform_buffer_allocation)?;

        state.device.destroy_buffer(first_phase_data_buffer, None);
        state.allocator().free(first_phase_data_buffer_allocation)?;
        state
            .device
            .destroy_buffer(first_phase_data_cpu_buffer, None);
        state
            .allocator()
            .free(first_phase_data_cpu_buffer_allocation)?;

        state
            .device
            .destroy_buffer(second_phase_material_data_buffer, None);
        state
            .allocator()
            .free(second_phase_material_data_buffer_allocation)?;
        state
            .device
            .destroy_buffer(second_phase_material_data_cpu_buffer, None);
        state
            .allocator()
            .free(second_phase_material_data_cpu_buffer_allocation)?;

        state.device.destroy_buffer(second_phase_data_buffer, None);
        state
            .allocator()
            .free(second_phase_data_buffer_allocation)?;
        state
            .device
            .destroy_buffer(second_phase_data_cpu_buffer, None);
        state
            .allocator()
            .free(second_phase_data_cpu_buffer_allocation)?;

        state.device.destroy_pipeline(first_phase_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(first_phase_pipeline_layout, None);
        state
            .device
            .destroy_pipeline(second_phase_material_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(second_phase_material_pipeline_layout, None);
        state.device.destroy_pipeline(second_phase_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(second_phase_pipeline_layout, None);
    }
    base_color_texture.destroy(state);
    metallic_texture.destroy(state);
    roughness_texture.destroy(state);
    normal_texture.destroy(state);

    Ok(())
}
