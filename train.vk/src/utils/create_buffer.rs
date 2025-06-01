use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::vulkan_state::VulkanState;

pub fn create_uniform_buffer(
    state: &mut VulkanState,
    size: u64,
) -> Result<(vk::Buffer, Allocation)> {
    // Create uniform buffer
    let uniform_buffer_create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let uniform_buffer = unsafe {
        state
            .device
            .create_buffer(&uniform_buffer_create_info, None)?
    };
    // Allocate memory for the uniform buffer
    let uniform_buffer_requirements =
        unsafe { state.device.get_buffer_memory_requirements(uniform_buffer) };
    let uniform_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
        name: "uniform buffer",
        requirements: uniform_buffer_requirements,
        location: MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    // Bind the uniform buffer memory
    unsafe {
        state.device.bind_buffer_memory(
            uniform_buffer,
            uniform_buffer_allocation.memory(),
            uniform_buffer_allocation.offset(),
        )?;
    }

    Ok((uniform_buffer, uniform_buffer_allocation))
}

pub fn create_storage_buffer(
    state: &mut VulkanState,
    size: u64,
) -> Result<(vk::Buffer, Allocation)> {
    // Create storage buffer
    let storage_buffer_create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let storage_buffer = unsafe {
        state
            .device
            .create_buffer(&storage_buffer_create_info, None)?
    };
    // Allocate memory for the storage buffer
    let storage_buffer_requirements =
        unsafe { state.device.get_buffer_memory_requirements(storage_buffer) };
    let storage_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
        name: "storage buffer",
        requirements: storage_buffer_requirements,
        location: MemoryLocation::GpuOnly,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    // Bind the storage buffer memory
    unsafe {
        state.device.bind_buffer_memory(
            storage_buffer,
            storage_buffer_allocation.memory(),
            storage_buffer_allocation.offset(),
        )?;
    }

    Ok((storage_buffer, storage_buffer_allocation))
}

pub fn create_cpu_storage_buffer(
    state: &mut VulkanState,
    size: u64,
) -> Result<(vk::Buffer, Allocation)> {
    // Create storage buffer
    let storage_buffer_create_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let storage_buffer = unsafe {
        state
            .device
            .create_buffer(&storage_buffer_create_info, None)?
    };
    // Allocate memory for the storage buffer
    let storage_buffer_requirements =
        unsafe { state.device.get_buffer_memory_requirements(storage_buffer) };
    let storage_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
        name: "storage buffer",
        requirements: storage_buffer_requirements,
        location: MemoryLocation::GpuToCpu,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    // Bind the storage buffer memory
    unsafe {
        state.device.bind_buffer_memory(
            storage_buffer,
            storage_buffer_allocation.memory(),
            storage_buffer_allocation.offset(),
        )?;
    }

    Ok((storage_buffer, storage_buffer_allocation))
}

pub fn create_storage_buffer_with_data(
    state: &mut VulkanState,
    data: &[u8],
) -> Result<(vk::Buffer, Allocation)> {
    let buffer_size = data.len() as u64;

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
    let mut staging_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
        name: "staging buffer",
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
    let map_data = staging_buffer_allocation
        .mapped_slice_mut()
        .ok_or_else(|| {
            panic!("Failed to map staging buffer memory");
        })?;
    map_data[0..buffer_size as usize].copy_from_slice(bytemuck::cast_slice(&data));

    // Create storage buffer
    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let storage_buffer = unsafe { state.device.create_buffer(&buffer_create_info, None)? };

    // Allocate memory for the storage buffer
    let storage_buffer_requirements =
        unsafe { state.device.get_buffer_memory_requirements(storage_buffer) };
    let storage_buffer_allocation = state.allocator().allocate(&AllocationCreateDesc {
        name: "storage buffer",
        requirements: storage_buffer_requirements,
        location: MemoryLocation::GpuOnly,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;

    // Bind the storage buffer memory
    unsafe {
        state.device.bind_buffer_memory(
            storage_buffer,
            storage_buffer_allocation.memory(),
            storage_buffer_allocation.offset(),
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
                .dst_buffer(storage_buffer)
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

    // Return the storage buffer and its memory
    Ok((storage_buffer, storage_buffer_allocation))
}
