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
