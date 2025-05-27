//! Based on an RTXNS ShaderTraining sample

use anyhow::Result;
use ash::vk;
use rand::prelude::*;

use crate::network::TrainedNetwork;
use crate::utils::create_compute_pipeline;
use crate::{
    network::Network,
    utils::{
        create_cpu_storage_buffer, create_storage_buffer, create_storage_buffer_with_data,
        create_uniform_buffer,
    },
    vulkan_state::VulkanState,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBuffer {
    weight_offsets: [u32; 4],
    bias_offsets: [u32; 4],
    batch_size: u32,
    params_size: u32,
    learning_rate: f32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    seed: u64,
    current_step: u32,
    _padding: u32,
}

pub fn train(state: &mut VulkanState, epochs: u32) -> Result<()> {
    let batch_size = 1 << 16;
    let batch_count = 100;
    let learning_rate = 0.01;
    let dimensions = [5 * 6, 32, 32, 32, 4];

    // Create network
    let network = Network::from_dimensions(&state.cooperative_vector_fn, &dimensions)?;
    let total_params_count = network.data.len() as u64 / std::mem::size_of::<half::f16>() as u64;

    // === Initialize buffers and pipelines ===

    let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<PushConstants>() as u32)];

    // Create uniform buffer and other buffers
    let (uniform_buffer, mut uniform_buffer_allocation) =
        create_uniform_buffer(state, std::mem::size_of::<UniformBuffer>() as u64)?;
    let (network_params_buffer, network_params_buffer_allocation) =
        create_storage_buffer_with_data(state, &network.data)?;
    let (network_params_float_buffer, network_params_float_buffer_allocation) =
        create_storage_buffer(
            state,
            total_params_count * std::mem::size_of::<f32>() as u64,
        )?;
    let (gradient_buffer, gradient_buffer_allocation) = create_storage_buffer(
        state,
        (total_params_count * std::mem::size_of::<half::f16>() as u64).div_ceil(4) * 4,
    )?;
    let (moment_1_buffer, moment_1_buffer_allocation) = create_storage_buffer(
        state,
        total_params_count * std::mem::size_of::<f32>() as u64,
    )?;
    let (moment_2_buffer, moment_2_buffer_allocation) = create_storage_buffer(
        state,
        total_params_count * std::mem::size_of::<f32>() as u64,
    )?;
    let (network_params_cpu_buffer, network_params_cpu_buffer_allocation) =
        create_cpu_storage_buffer(
            state,
            total_params_count * std::mem::size_of::<half::f16>() as u64,
        )?;

    // Create init descriptor set layout
    let init_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // network params float
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // moment 1 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // moment 2 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
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
    let init_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(5),
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
    let init_descriptor_set = {
        let set_layouts = [init_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(init_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    // Create init pipeline
    let (init_pipeline, init_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/disney_rtxns/init.comp.spv"
            )),
            &[init_descriptor_set_layout],
            &push_constant_ranges,
        )?
    };
    // Update init descriptor set
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBuffer>() as u64)];
        let network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_params_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let network_params_float_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_params_float_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<f32>() as u64)];
        let gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(gradient_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let moment_1_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(moment_1_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<f32>() as u64)];
        let moment_2_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(moment_2_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<f32>() as u64)];
        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(init_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // network params
            vk::WriteDescriptorSet::default()
                .dst_set(init_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&network_params_buffer_info),
            // network params float
            vk::WriteDescriptorSet::default()
                .dst_set(init_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&network_params_float_buffer_info),
            // gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(init_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gradient_buffer_info),
            // moment 1 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(init_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&moment_1_buffer_info),
            // moment 2 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(init_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&moment_2_buffer_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    // Create train descriptor set layout
    let train_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // gradient buffer
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
    // Create train descriptor pool
    let train_descriptor_pool = {
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
    // Create train descriptor set
    let train_descriptor_set = {
        let set_layouts = [train_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(train_descriptor_pool)
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
            .buffer(uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBuffer>() as u64)];
        let network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_params_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(gradient_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(train_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // network params
            vk::WriteDescriptorSet::default()
                .dst_set(train_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&network_params_buffer_info),
            // gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(train_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gradient_buffer_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }
    // Create train pipeline
    let (train_pipeline, train_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/disney_rtxns/train.comp.spv"
            )),
            &[train_descriptor_set_layout],
            &push_constant_ranges,
        )?
    };

    // Create optimization descriptor set layout
    let optimization_descriptor_set_layout = {
        let bindings = [
            // uniform buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // network params
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // network params float
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // gradient buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // moment 1 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // moment 2 buffer
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
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
    let optimization_descriptor_pool = {
        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(5),
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
    let optimization_descriptor_set = {
        let set_layouts = [optimization_descriptor_set_layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(optimization_descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe {
            state
                .device
                .allocate_descriptor_sets(&allocate_info)
                .expect("Allocate descriptor set failed")
        }
    }[0];
    // Create optimization pipeline
    let (optimize_pipeline, optimize_pipeline_layout) = {
        create_compute_pipeline(
            state,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/disney_rtxns/adam.comp.spv"
            )),
            &[optimization_descriptor_set_layout],
            &push_constant_ranges,
        )?
    };
    // Update optimization descriptor set
    {
        let uniform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBuffer>() as u64)];
        let network_params_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_params_buffer)
            .offset(0)
            .range(network.data.len() as u64)];
        let network_params_float_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(network_params_float_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<f32>() as u64)];
        let gradient_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(gradient_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<half::f16>() as u64)];
        let moment_1_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(moment_1_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<f32>() as u64)];
        let moment_2_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(moment_2_buffer)
            .offset(0)
            .range(total_params_count * std::mem::size_of::<f32>() as u64)];
        let descriptor_writes = [
            // uniform buffer
            vk::WriteDescriptorSet::default()
                .dst_set(optimization_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_buffer_info),
            // network params
            vk::WriteDescriptorSet::default()
                .dst_set(optimization_descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&network_params_buffer_info),
            // network params float
            vk::WriteDescriptorSet::default()
                .dst_set(optimization_descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&network_params_float_buffer_info),
            // gradient buffer
            vk::WriteDescriptorSet::default()
                .dst_set(optimization_descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gradient_buffer_info),
            // moment 1 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(optimization_descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&moment_1_buffer_info),
            // moment 2 buffer
            vk::WriteDescriptorSet::default()
                .dst_set(optimization_descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&moment_2_buffer_info),
        ];
        unsafe { state.device.update_descriptor_sets(&descriptor_writes, &[]) };
    }

    // === convert weights ===

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

    // Update uniform buffer
    let uniform_data = UniformBuffer {
        weight_offsets: [
            network.weight_offsets[0],
            network.weight_offsets[1],
            network.weight_offsets[2],
            network.weight_offsets[3],
        ],
        bias_offsets: [
            network.bias_offsets[0],
            network.bias_offsets[1],
            network.bias_offsets[2],
            network.bias_offsets[3],
        ],
        batch_size,
        params_size: total_params_count as u32,
        learning_rate,
        _padding: 0,
    };
    uniform_buffer_allocation
        .mapped_slice_mut()
        .expect("Failed to map uniform buffer")[0..std::mem::size_of::<UniformBuffer>()]
        .copy_from_slice(bytemuck::bytes_of(&uniform_data));

    unsafe {
        state
            .device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
        state.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            init_pipeline,
        );
        state.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            init_pipeline_layout,
            0,
            &[init_descriptor_set],
            &[],
        );
        state.device.cmd_push_constants(
            command_buffer,
            init_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&PushConstants {
                seed: 0,
                current_step: 0,
                _padding: 0,
            }),
        );
        state
            .device
            .cmd_dispatch(command_buffer, total_params_count.div_ceil(32) as u32, 1, 1);
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

    // === training loop ===

    let mut rng = rand::rng();

    // training loop
    for i in 0..epochs {
        println!("Epoch {}/{}", i + 1, epochs);

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
                    train_pipeline,
                );
                state.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    train_pipeline_layout,
                    0,
                    &[train_descriptor_set],
                    &[],
                );
                state.device.cmd_push_constants(
                    command_buffer,
                    init_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&PushConstants {
                        seed,
                        current_step: i * batch_count + j,
                        _padding: 0,
                    }),
                );
                state
                    .device
                    .cmd_dispatch(command_buffer, batch_size.div_ceil(32), 1, 1);
            }
            let buffer_memory_barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .buffer(gradient_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
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
                    optimize_pipeline,
                );
                state.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    optimize_pipeline_layout,
                    0,
                    &[optimization_descriptor_set],
                    &[],
                );
                state.device.cmd_push_constants(
                    command_buffer,
                    init_pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&PushConstants {
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
            let buffer_memory_barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .buffer(gradient_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            let dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
            unsafe {
                state
                    .device
                    .cmd_pipeline_barrier2(command_buffer, &dependency_info);
            }
        }

        if (epochs % 100 == 0 || i == epochs - 1) && i > 0 {
            // Barrier to ensure all writes are visible before copying
            let buffer_memory_barriers = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .buffer(network_params_buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            let dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&buffer_memory_barriers);
            unsafe {
                state
                    .device
                    .cmd_pipeline_barrier2(command_buffer, &dependency_info);
            }

            // copy params to CPU buffer
            unsafe {
                state.device.cmd_copy_buffer(
                    command_buffer,
                    network_params_buffer,
                    network_params_cpu_buffer,
                    &[vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(0)
                        .size(total_params_count * std::mem::size_of::<half::f16>() as u64)],
                );
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

        if (epochs % 100 == 0 || i == epochs - 1) && i > 0 {
            // copy data
            let data = network_params_cpu_buffer_allocation
                .mapped_slice()
                .expect("Failed to map network params CPU buffer")
                .to_vec();

            let trained_network = TrainedNetwork::from_data(
                &state.cooperative_vector_fn,
                &data,
                &network.weight_offsets,
                &network.bias_offsets,
                &dimensions,
            )?;
            trained_network.save_network("./network/disney-rtxns.json")?;
        }
    }

    // destroy resources
    unsafe {
        state
            .device
            .destroy_descriptor_set_layout(train_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_pool(train_descriptor_pool, None);
        state.device.destroy_pipeline(train_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(train_pipeline_layout, None);

        state
            .device
            .destroy_descriptor_set_layout(optimization_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_pool(optimization_descriptor_pool, None);
        state.device.destroy_pipeline(optimize_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(optimize_pipeline_layout, None);

        state
            .device
            .destroy_descriptor_set_layout(init_descriptor_set_layout, None);
        state
            .device
            .destroy_descriptor_pool(init_descriptor_pool, None);
        state.device.destroy_pipeline(init_pipeline, None);
        state
            .device
            .destroy_pipeline_layout(init_pipeline_layout, None);

        state.allocator().free(uniform_buffer_allocation)?;
        state.allocator().free(network_params_buffer_allocation)?;
        state
            .allocator()
            .free(network_params_float_buffer_allocation)?;
        state.allocator().free(gradient_buffer_allocation)?;
        state.allocator().free(moment_1_buffer_allocation)?;
        state.allocator().free(moment_2_buffer_allocation)?;
        state
            .allocator()
            .free(network_params_cpu_buffer_allocation)?;
        state.device.destroy_buffer(uniform_buffer, None);
        state.device.destroy_buffer(network_params_buffer, None);
        state
            .device
            .destroy_buffer(network_params_float_buffer, None);
        state.device.destroy_buffer(gradient_buffer, None);
        state.device.destroy_buffer(moment_1_buffer, None);
        state.device.destroy_buffer(moment_2_buffer, None);
        state.device.destroy_buffer(network_params_cpu_buffer, None);
    }

    Ok(())
}
