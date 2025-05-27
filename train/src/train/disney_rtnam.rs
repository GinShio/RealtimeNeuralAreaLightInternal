//! Real-Time Neural Appearance Model for Disney BRDF model

use std::io::Write;
use std::path::Path;

use anyhow::Result;
use ash::vk;
use rand::prelude::*;

use crate::{
    network::{Network, TrainedNetwork},
    utils::{
        create_compute_pipeline, create_cpu_storage_buffer, create_storage_buffer,
        create_storage_buffer_with_data, create_uniform_buffer, load_glb_texture,
    },
    vulkan_state::VulkanState,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FirstPhaseUniformBuffer {
    encoder_weight_offsets_1: [u32; 4],
    encoder_weight_offsets_2: [u32; 4],
    encoder_bias_offsets_1: [u32; 4],
    encoder_bias_offsets_2: [u32; 4],

    decoder_weight_offsets_1: [u32; 4],
    decoder_weight_offsets_2: [u32; 4],
    decoder_bias_offsets_1: [u32; 4],
    decoder_bias_offsets_2: [u32; 4],

    batch_size: u32,
    encoder_params_size: u32,
    decoder_params_size: u32,
    learning_rate: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FirstPhasePushConstants {
    seed: u64,
    current_step: u32,
    _padding: u32,
}

pub fn train(
    state: &mut VulkanState,
    path: impl AsRef<Path>,
    epochs: u32,
    texture_size: u32,
) -> Result<()> {
    let batch_size = 1 << 16;
    let batch_count = 100;
    let learning_rate = 0.01;

    // === Create networks ===

    // input: base color (3), roughness, metallic, normal (3)
    // output: latent vector (8)
    let encoder_dimensions = [(8, 64), (64, 64), (64, 64), (64, 64), (64, 8)];

    // input: latent vector (8)
    // middle: latent vector (8), transform frame (12)
    // output: BRDF (3)
    let decoder_dimensions = [(8, 12), (8 + 12, 64), (64, 64), (64, 64), (64, 3)];

    // Create encoder network
    let encoder_network =
        Network::from_dimensions(&state.cooperative_vector_fn, &encoder_dimensions)?;
    let encoder_total_params_count =
        encoder_network.data.len() as u64 / std::mem::size_of::<half::f16>() as u64;

    // Create decoder network
    let decoder_network =
        Network::from_dimensions(&state.cooperative_vector_fn, &decoder_dimensions)?;
    let decoder_total_params_count =
        decoder_network.data.len() as u64 / std::mem::size_of::<half::f16>() as u64;

    // total parameters counts of two networks
    let total_params_count = encoder_total_params_count + decoder_total_params_count;

    // Total pixels of latent texture
    let total_latent_texture_pixel_count = {
        let mut pixel_count = 0;
        let mut width = texture_size as u64;
        while width > 1 {
            pixel_count += width * width;
            width /= 2;
        }
        pixel_count
    };

    // === Load texture data ===
    let mut glb_textures = load_glb_texture(state, path, texture_size);

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

    let first_phase_push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<FirstPhasePushConstants>() as u32)];

    // Create uniform buffers
    let (first_phase_uniform_buffer, mut first_phase_uniform_buffer_allocation) =
        create_uniform_buffer(state, std::mem::size_of::<FirstPhaseUniformBuffer>() as u64)?;

    // Create encoder network buffers
    let (encoder_network_params_buffer, encoder_network_params_buffer_allocation) =
        create_storage_buffer_with_data(state, &encoder_network.data)?;
    let (encoder_network_params_float_buffer, encoder_network_params_float_buffer_allocation) =
        create_storage_buffer(
            state,
            encoder_total_params_count * std::mem::size_of::<f32>() as u64,
        )?;
    let (encoder_gradient_buffer, encoder_gradient_buffer_allocation) = create_storage_buffer(
        state,
        (encoder_total_params_count * std::mem::size_of::<half::f16>() as u64).div_ceil(4) * 4,
    )?;
    let (encoder_moment_1_buffer, encoder_moment_1_buffer_allocation) = create_storage_buffer(
        state,
        encoder_total_params_count * std::mem::size_of::<f32>() as u64,
    )?;
    let (encoder_moment_2_buffer, encoder_moment_2_buffer_allocation) = create_storage_buffer(
        state,
        encoder_total_params_count * std::mem::size_of::<f32>() as u64,
    )?;

    // Create decoder network buffers
    let (decoder_network_params_buffer, decoder_network_params_buffer_allocation) =
        create_storage_buffer_with_data(state, &decoder_network.data)?;
    let (decoder_network_params_float_buffer, decoder_network_params_float_buffer_allocation) =
        create_storage_buffer(
            state,
            decoder_total_params_count * std::mem::size_of::<f32>() as u64,
        )?;
    let (decoder_gradient_buffer, decoder_gradient_buffer_allocation) = create_storage_buffer(
        state,
        (decoder_total_params_count * std::mem::size_of::<half::f16>() as u64).div_ceil(4) * 4,
    )?;
    let (decoder_moment_1_buffer, decoder_moment_1_buffer_allocation) = create_storage_buffer(
        state,
        decoder_total_params_count * std::mem::size_of::<f32>() as u64,
    )?;
    let (decoder_moment_2_buffer, decoder_moment_2_buffer_allocation) = create_storage_buffer(
        state,
        decoder_total_params_count * std::mem::size_of::<f32>() as u64,
    )?;
    let (decoder_network_params_cpu_buffer, decoder_network_params_cpu_buffer_allocation) =
        create_cpu_storage_buffer(
            state,
            decoder_total_params_count * std::mem::size_of::<half::f16>() as u64,
        )?;

    // Create latent texture buffers
    let (latent_texture_network_params_buffer, latent_texture_network_params_buffer_allocation) =
        create_storage_buffer(
            state,
            (total_latent_texture_pixel_count * 8 * std::mem::size_of::<half::f16>() as u64)
                .div_ceil(4)
                * 4,
        )?;
    let (
        latent_texture_network_params_float_buffer,
        latent_texture_network_params_float_buffer_allocation,
    ) = create_storage_buffer(
        state,
        total_latent_texture_pixel_count * 8 * std::mem::size_of::<f32>() as u64,
    )?;
    let (latent_texture_gradient_buffer, latent_texture_gradient_buffer_allocation) =
        create_storage_buffer(
            state,
            (total_latent_texture_pixel_count * 8 * std::mem::size_of::<half::f16>() as u64)
                .div_ceil(4)
                * 4,
        )?;
    let (latent_texture_moment_1_buffer, latent_texture_moment_1_buffer_allocation) =
        create_storage_buffer(
            state,
            total_latent_texture_pixel_count * 8 * std::mem::size_of::<f32>() as u64,
        )?;
    let (latent_texture_moment_2_buffer, latent_texture_moment_2_buffer_allocation) =
        create_storage_buffer(
            state,
            total_latent_texture_pixel_count * 8 * std::mem::size_of::<f32>() as u64,
        )?;
    let (
        latent_texture_network_params_cpu_buffer,
        latent_texture_network_params_cpu_buffer_allocation,
    ) = create_cpu_storage_buffer(
        state,
        total_latent_texture_pixel_count * 8 * std::mem::size_of::<half::f16>() as u64,
    )?;

    // === Create 1st phase descriptors ===

    // Create init descriptor set layout
    let first_phase_init_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder network params float
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder moment 1 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder moment 2 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder network params float
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder moment 1 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder moment 2 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
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
    // Create init descriptor pool
    let first_phase_init_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10),
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
    // Create init descriptor set
    let first_phase_init_descriptor_set = {
        let set_layouts = [first_phase_init_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(first_phase_init_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    // Update init descriptor set
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(first_phase_uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<FirstPhaseUniformBuffer>() as u64)];

        let encoder_network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_network_params_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let encoder_network_params_float_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_network_params_float_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let encoder_gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_gradient_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let encoder_moment_1_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_moment_1_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let encoder_moment_2_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_moment_2_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<f32>() as u64)];

        let decoder_network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_network_params_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let decoder_network_params_float_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_network_params_float_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let decoder_gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_gradient_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let decoder_moment_1_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_moment_1_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let decoder_moment_2_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_moment_2_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<f32>() as u64)];

        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // encoder network params
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_network_params_buffer_info),
            // encoder network params float
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_network_params_float_buffer_info),
            // encoder gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_gradient_buffer_info),
            // encoder moment 1 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_moment_1_buffer_info),
            // encoder moment 2 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_moment_2_buffer_info),
            // decoder network params
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_network_params_buffer_info),
            // decoder network params float
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_network_params_float_buffer_info),
            // decoder gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_gradient_buffer_info),
            // decoder moment 1 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_moment_1_buffer_info),
            // decoder moment 2 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_init_descriptor_set)
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_moment_2_buffer_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    // Create train descriptor set layout
    let first_phase_train_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder_network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder_gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder_network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder_gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // base color texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // roughness metallic texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // normal texture
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
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
    // Create train descriptor pool
    let first_phase_train_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(3),
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
    // Create train descriptor set
    let first_phase_train_descriptor_set = {
        let set_layouts = [first_phase_train_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(first_phase_train_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    // Update train descriptor set
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(first_phase_uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<FirstPhaseUniformBuffer>() as u64)];

        let encoder_network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_network_params_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let encoder_gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_gradient_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];

        let decoder_network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_network_params_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let decoder_gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_gradient_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];

        let base_color_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(glb_textures.base_color.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let roughness_metallic_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(glb_textures.metallic_roughness.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let normal_texture_info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(glb_textures.normal.image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // encoder network params
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_network_params_buffer_info),
            // encoder gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_gradient_buffer_info),
            // decoder network params
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_network_params_buffer_info),
            // decoder gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_gradient_buffer_info),
            // base color texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&base_color_texture_info),
            // roughness metallic texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&roughness_metallic_texture_info),
            // normal texture
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_train_descriptor_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&normal_texture_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    // Create optimization descriptor set layout
    let first_phase_optimization_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder network params float
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder moment 1 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // encoder moment 2 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder network params float
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder moment 1 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // decoder moment 2 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
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
    // Create optimization descriptor pool
    let first_phase_optimization_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10),
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
    // Create optimization descriptor set
    let first_phase_optimization_descriptor_set = {
        let set_layouts = [first_phase_optimization_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(first_phase_optimization_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    // Update optimization descriptor set
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(first_phase_uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<FirstPhaseUniformBuffer>() as u64)];

        let encoder_network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_network_params_buffer)
            .offset(0)
            .range(encoder_network.data.len() as u64)];
        let encoder_network_params_float_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_network_params_float_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let encoder_gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_gradient_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let encoder_moment_1_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_moment_1_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let encoder_moment_2_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(encoder_moment_2_buffer)
            .offset(0)
            .range(encoder_total_params_count * std::mem::size_of::<f32>() as u64)];

        let decoder_network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_network_params_buffer)
            .offset(0)
            .range(decoder_network.data.len() as u64)];
        let decoder_network_params_float_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_network_params_float_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let decoder_gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_gradient_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let decoder_moment_1_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_moment_1_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<f32>() as u64)];
        let decoder_moment_2_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(decoder_moment_2_buffer)
            .offset(0)
            .range(decoder_total_params_count * std::mem::size_of::<f32>() as u64)];

        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // encoder network params
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_network_params_buffer_info),
            // encoder network params float
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_network_params_float_buffer_info),
            // encoder gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_gradient_buffer_info),
            // encoder moment 1 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_moment_1_buffer_info),
            // encoder moment 2 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&encoder_moment_2_buffer_info),
            // decoder network params
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_network_params_buffer_info),
            // decoder network params float
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_network_params_float_buffer_info),
            // decoder gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_gradient_buffer_info),
            // decoder moment 1 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_moment_1_buffer_info),
            // decoder moment 2 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(first_phase_optimization_descriptor_set)
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&decoder_moment_2_buffer_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    // === Create 1st phase pipeline ===

    // Create init pipeline
    let (first_phase_init_pipeline, first_phase_init_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/disney_rtnam/init_1st.comp.spv"
            )),
            &[first_phase_init_descriptor_set_layout],
            &first_phase_push_constant_ranges,
        )?
    };

    // Create train pipeline
    let (first_phase_train_pipeline, first_phase_train_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/disney_rtxns/train.comp.spv"
            )),
            &[first_phase_train_descriptor_set_layout],
            &first_phase_push_constant_ranges,
        )?
    };

    // Create optimization pipeline
    let (first_phase_optimize_pipeline, first_phase_optimize_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/disney_rtxns/adam.comp.spv"
            )),
            &[first_phase_optimization_descriptor_set_layout],
            &first_phase_push_constant_ranges,
        )?
    };

    // === 1st Phase ===

    // Update uniform buffer
    let uniform_data = FirstPhaseUniformBuffer {
        encoder_weight_offsets_1: [
            encoder_network.weight_offsets[0],
            encoder_network.weight_offsets[1],
            encoder_network.weight_offsets[2],
            encoder_network.weight_offsets[3],
        ],
        encoder_weight_offsets_2: [encoder_network.weight_offsets[4], 0, 0, 0],
        encoder_bias_offsets_1: [
            encoder_network.bias_offsets[0],
            encoder_network.bias_offsets[1],
            encoder_network.bias_offsets[2],
            encoder_network.bias_offsets[3],
        ],
        encoder_bias_offsets_2: [encoder_network.bias_offsets[4], 0, 0, 0],

        decoder_weight_offsets_1: [
            decoder_network.weight_offsets[0],
            decoder_network.weight_offsets[1],
            decoder_network.weight_offsets[2],
            decoder_network.weight_offsets[3],
        ],
        decoder_weight_offsets_2: [decoder_network.weight_offsets[4], 0, 0, 0],
        decoder_bias_offsets_1: [
            decoder_network.bias_offsets[0],
            decoder_network.bias_offsets[1],
            decoder_network.bias_offsets[2],
            decoder_network.bias_offsets[3],
        ],
        decoder_bias_offsets_2: [decoder_network.bias_offsets[4], 0, 0, 0],

        batch_size,
        encoder_params_size: encoder_total_params_count as u32,
        decoder_params_size: decoder_total_params_count as u32,
        learning_rate,
    };
    first_phase_uniform_buffer_allocation
        .mapped_slice_mut()
        .expect("Failed to map uniform buffer")[0..std::mem::size_of::<FirstPhaseUniformBuffer>()]
        .copy_from_slice(bytemuck::bytes_of(&uniform_data));

    unsafe {
        let command_buffer = state.begin_single_time_commands();

        state.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            first_phase_init_pipeline,
        );
        state.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            first_phase_init_pipeline_layout,
            0,
            &[first_phase_init_descriptor_set],
            &[],
        );
        state.device.cmd_push_constants(
            command_buffer,
            first_phase_init_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&FirstPhasePushConstants {
                seed: 0,
                current_step: 0,
                _padding: 0,
            }),
        );
        state
            .device
            .cmd_dispatch(command_buffer, total_params_count.div_ceil(32) as u32, 1, 1);

        state.end_single_time_commands(command_buffer);
    }
    // Create command buffer for training
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(state.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe {
        state
            .device
            .allocate_command_buffers(&command_buffer_allocate_info)?
    }[0];

    // training loop
    let mut rng = rand::rng();
    for i in 0..epochs {
        print!("\r  First Phase Epoch {}/{}", i + 1, epochs);
        std::io::stdout().flush().expect("Failed to flush stdout");

        // begin command buffer
        unsafe {
            state
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
        }

        for j in 0..batch_count {
            let seed = rng.random();

            // training pass
            unsafe {
                state.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    first_phase_train_pipeline,
                );
                state.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    first_phase_train_pipeline_layout,
                    0,
                    &[first_phase_train_descriptor_set],
                    &[],
                );
                state.device.cmd_push_constants(
                    command_buffer,
                    first_phase_train_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&FirstPhasePushConstants {
                        seed,
                        current_step: i * batch_count + j,
                        _padding: 0,
                    }),
                );
                state
                    .device
                    .cmd_dispatch(command_buffer, batch_size.div_ceil(32), 1, 1);
            }
            let buffer_memory_barriers = [
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(encoder_gradient_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(decoder_gradient_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            ];
            let dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
            unsafe {
                state
                    .device
                    .cmd_pipeline_barrier2(command_buffer, &dependency_info);
            }

            // optimization pass
            unsafe {
                state.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    first_phase_optimize_pipeline,
                );
                state.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    first_phase_optimize_pipeline_layout,
                    0,
                    &[first_phase_optimization_descriptor_set],
                    &[],
                );
                state.device.cmd_push_constants(
                    command_buffer,
                    first_phase_optimize_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&FirstPhasePushConstants {
                        seed,
                        current_step: i * batch_count + j,
                        _padding: 0,
                    }),
                );
                state.device.cmd_dispatch(
                    command_buffer,
                    total_params_count.div_ceil(32) as u32,
                    1,
                    1,
                );
            }
            let buffer_memory_barriers = [
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(encoder_network_params_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(encoder_gradient_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(decoder_network_params_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .buffer(decoder_gradient_buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            ];
            let dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
            unsafe {
                state
                    .device
                    .cmd_pipeline_barrier2(command_buffer, &dependency_info);
            }
        }

        // end command buffer
        unsafe {
            state.device.end_command_buffer(command_buffer)?;
        }

        // submit command buffer
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe {
            state
                .device
                .queue_submit(state.queue, &[submit_info], vk::Fence::null())?;
            state.device.queue_wait_idle(state.queue)?;
        }
        // reset command buffer
        unsafe {
            state
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
    }

    println!("\r  First Phase Epoch {}/{} - Done", epochs, epochs);

    // === 2nd Phase ===

    // todo

    // === Cleanup ===

    // destroy first phase descriptor sets and pipelines
    unsafe {
        state
            .device
            .destroy_descriptor_pool(first_phase_init_descriptor_pool, None);
        state
            .device
            .destroy_descriptor_set_layout(first_phase_init_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_pool(first_phase_train_descriptor_pool, None);
        state
            .device
            .destroy_descriptor_set_layout(first_phase_train_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_pool(first_phase_optimization_descriptor_pool, None);
        state
            .device
            .destroy_descriptor_set_layout(first_phase_optimization_descriptor_set_layout, None);
    }

    // destroy first phase pipelines
    unsafe {
        state
            .device
            .destroy_pipeline(first_phase_init_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(first_phase_init_pipeline_layout, None);
        state
            .device
            .destroy_pipeline(first_phase_train_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(first_phase_train_pipeline_layout, None);
        state
            .device
            .destroy_pipeline(first_phase_optimize_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(first_phase_optimize_pipeline_layout, None);
    }

    // Destroy buffers
    unsafe {
        state
            .device
            .destroy_buffer(first_phase_uniform_buffer, None);
        state
            .allocator()
            .free(first_phase_uniform_buffer_allocation)
            .expect("Failed to free first phase uniform buffer");

        state
            .device
            .destroy_buffer(encoder_network_params_buffer, None);
        state
            .allocator()
            .free(encoder_network_params_buffer_allocation)
            .expect("Failed to free encoder network params buffer");
        state
            .device
            .destroy_buffer(encoder_network_params_float_buffer, None);
        state
            .allocator()
            .free(encoder_network_params_float_buffer_allocation)
            .expect("Failed to free encoder network params float buffer");
        state.device.destroy_buffer(encoder_gradient_buffer, None);
        state
            .allocator()
            .free(encoder_gradient_buffer_allocation)
            .expect("Failed to free encoder gradient buffer");
        state.device.destroy_buffer(encoder_moment_1_buffer, None);
        state
            .allocator()
            .free(encoder_moment_1_buffer_allocation)
            .expect("Failed to free encoder moment 1 buffer");
        state.device.destroy_buffer(encoder_moment_2_buffer, None);
        state
            .allocator()
            .free(encoder_moment_2_buffer_allocation)
            .expect("Failed to free encoder moment 2 buffer");

        state
            .device
            .destroy_buffer(decoder_network_params_buffer, None);
        state
            .allocator()
            .free(decoder_network_params_buffer_allocation)
            .expect("Failed to free decoder network params buffer");
        state
            .device
            .destroy_buffer(decoder_network_params_float_buffer, None);
        state
            .allocator()
            .free(decoder_network_params_float_buffer_allocation)
            .expect("Failed to free decoder network params float buffer");
        state.device.destroy_buffer(decoder_gradient_buffer, None);
        state
            .allocator()
            .free(decoder_gradient_buffer_allocation)
            .expect("Failed to free decoder gradient buffer");
        state.device.destroy_buffer(decoder_moment_1_buffer, None);
        state
            .allocator()
            .free(decoder_moment_1_buffer_allocation)
            .expect("Failed to free decoder moment 1 buffer");
        state.device.destroy_buffer(decoder_moment_2_buffer, None);
        state
            .allocator()
            .free(decoder_moment_2_buffer_allocation)
            .expect("Failed to free decoder moment 2 buffer");
        state
            .device
            .destroy_buffer(decoder_network_params_cpu_buffer, None);
        state
            .allocator()
            .free(decoder_network_params_cpu_buffer_allocation)
            .expect("Failed to free decoder network params CPU buffer");

        state
            .device
            .destroy_buffer(latent_texture_network_params_buffer, None);
        state
            .allocator()
            .free(latent_texture_network_params_buffer_allocation)
            .expect("Failed to free latent texture network params buffer");
        state
            .device
            .destroy_buffer(latent_texture_network_params_float_buffer, None);
        state
            .allocator()
            .free(latent_texture_network_params_float_buffer_allocation)
            .expect("Failed to free latent texture network params float buffer");
        state
            .device
            .destroy_buffer(latent_texture_gradient_buffer, None);
        state
            .allocator()
            .free(latent_texture_gradient_buffer_allocation)
            .expect("Failed to free latent texture gradient buffer");
        state
            .device
            .destroy_buffer(latent_texture_moment_1_buffer, None);
        state
            .allocator()
            .free(latent_texture_moment_1_buffer_allocation)
            .expect("Failed to free latent texture moment 1 buffer");
        state
            .device
            .destroy_buffer(latent_texture_moment_2_buffer, None);
        state
            .allocator()
            .free(latent_texture_moment_2_buffer_allocation)
            .expect("Failed to free latent texture moment 2 buffer");
        state
            .device
            .destroy_buffer(latent_texture_network_params_cpu_buffer, None);
        state
            .allocator()
            .free(latent_texture_network_params_cpu_buffer_allocation)
            .expect("Failed to free latent texture network params CPU buffer");
    }

    // Destroy sampler
    unsafe {
        state.device.destroy_sampler(sampler, None);
    }

    // Destroy textures
    glb_textures.base_color.destroy(state);
    glb_textures.metallic_roughness.destroy(state);
    glb_textures.normal.destroy(state);

    Ok(())
}
