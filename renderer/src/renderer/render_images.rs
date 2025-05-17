use anyhow::Result;
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};

use crate::renderer::{Renderer, vulkan_state::VulkanState};

/// A struct that holds the images used for rendering.
pub struct RenderImages {
    pub linear_scene_images: Vec<vk::Image>,
    pub linear_scene_image_views: Vec<vk::ImageView>,
    linear_scene_image_allocations: Vec<Allocation>,

    pub after_tone_mapping_images: Vec<vk::Image>,
    pub after_tone_mapping_image_views: Vec<vk::ImageView>,
    after_tone_mapping_image_allocations: Vec<Allocation>,
}
impl RenderImages {
    /// Creates a new instance of the RenderImages struct.
    pub fn new(state: &mut VulkanState) -> Result<Self> {
        // Create linear scene images
        let (linear_scene_images, linear_scene_image_views, linear_scene_image_allocations) = (0
            ..Renderer::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                // Create scene image
                let image_create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .mip_levels(1)
                    .array_layers(1)
                    .extent(vk::Extent3D {
                        width: state.swapchain.extent.width,
                        height: state.swapchain.extent.height,
                        depth: 1,
                    })
                    .samples(vk::SampleCountFlags::TYPE_1);
                let image = unsafe { state.device.create_image(&image_create_info, None) }.unwrap();

                // Allocate memory for the image
                let image_requirements =
                    unsafe { state.device.get_image_memory_requirements(image) };
                let image_allocation = state
                    .allocator()
                    .allocate(&AllocationCreateDesc {
                        name: "linear scene image",
                        requirements: image_requirements,
                        location: MemoryLocation::GpuOnly,
                        linear: false,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();
                unsafe {
                    state.device.bind_image_memory(
                        image,
                        image_allocation.memory(),
                        image_allocation.offset(),
                    )
                }
                .unwrap();

                // Create image view
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
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
                        .create_image_view(&image_view_create_info, None)
                        .expect("Failed to create image view")
                };

                // Create a command buffer
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(state.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let command_buffer = unsafe {
                    state
                        .device
                        .allocate_command_buffers(&command_buffer_allocate_info)
                        .unwrap()
                }[0];

                // Change image layout
                let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
                    .image(image)
                    .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2KHR::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2KHR::NONE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })];
                unsafe {
                    state
                        .device
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
                        .unwrap();
                    state.device.cmd_pipeline_barrier2(
                        command_buffer,
                        &vk::DependencyInfo::default()
                            .image_memory_barriers(&image_memory_barriers),
                    );
                    state.device.end_command_buffer(command_buffer).unwrap();
                }

                // Create a fence
                let fence_create_info = vk::FenceCreateInfo::default();
                let fence = unsafe { state.device.create_fence(&fence_create_info, None).unwrap() };

                // Submit the command buffer
                let buffers_for_submission = [command_buffer];
                let submit_info =
                    vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
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

                // Destroy the fence and command buffer
                unsafe {
                    state.device.destroy_fence(fence, None);
                    state
                        .device
                        .free_command_buffers(state.command_pool, &[command_buffer]);
                }

                (image, image_view, image_allocation)
            })
            .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

        // Create after tone mapping images
        let (
            after_tone_mapping_images,
            after_tone_mapping_image_views,
            after_tone_mapping_image_allocations,
        ) = (0..Renderer::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                // Create after tone mapping image
                let image_create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .mip_levels(1)
                    .array_layers(1)
                    .extent(vk::Extent3D {
                        width: state.swapchain.extent.width,
                        height: state.swapchain.extent.height,
                        depth: 1,
                    })
                    .samples(vk::SampleCountFlags::TYPE_1);
                let image = unsafe { state.device.create_image(&image_create_info, None) }.unwrap();

                // Allocate memory for the image
                let image_requirements =
                    unsafe { state.device.get_image_memory_requirements(image) };
                let image_allocation = state
                    .allocator()
                    .allocate(&AllocationCreateDesc {
                        name: "after tone mapping image",
                        requirements: image_requirements,
                        location: MemoryLocation::GpuOnly,
                        linear: false,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();
                unsafe {
                    state.device.bind_image_memory(
                        image,
                        image_allocation.memory(),
                        image_allocation.offset(),
                    )
                }
                .unwrap();

                // Create image view
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
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
                        .create_image_view(&image_view_create_info, None)
                        .expect("Failed to create image view")
                };

                // Create a command buffer
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(state.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let command_buffer = unsafe {
                    state
                        .device
                        .allocate_command_buffers(&command_buffer_allocate_info)
                        .unwrap()
                }[0];

                // Change image layout
                let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
                    .image(image)
                    .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2KHR::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2KHR::NONE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })];
                unsafe {
                    state
                        .device
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
                        .unwrap();
                    state.device.cmd_pipeline_barrier2(
                        command_buffer,
                        &vk::DependencyInfo::default()
                            .image_memory_barriers(&image_memory_barriers),
                    );
                    state.device.end_command_buffer(command_buffer).unwrap();
                }

                // Create a fence
                let fence_create_info = vk::FenceCreateInfo::default();
                let fence = unsafe { state.device.create_fence(&fence_create_info, None).unwrap() };

                // Submit the command buffer
                let buffers_for_submission = [command_buffer];
                let submit_info =
                    vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
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

                // Destroy the fence and command buffer
                unsafe {
                    state.device.destroy_fence(fence, None);
                    state
                        .device
                        .free_command_buffers(state.command_pool, &[command_buffer]);
                }

                (image, image_view, image_allocation)
            })
            .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

        Ok(Self {
            linear_scene_images,
            linear_scene_image_views,
            linear_scene_image_allocations,
            after_tone_mapping_images,
            after_tone_mapping_image_views,
            after_tone_mapping_image_allocations,
        })
    }

    /// Recreates the images.
    pub fn recreate(&mut self, state: &mut VulkanState) -> Result<()> {
        // Destroy old images
        self.destroy(state);

        // Create new images
        let new_images = Self::new(state)?;
        *self = new_images;

        Ok(())
    }

    /// Destroys the images.
    pub fn destroy(&mut self, state: &mut VulkanState) {
        unsafe {
            for image_view in &self.linear_scene_image_views {
                state.device.destroy_image_view(*image_view, None);
            }
            for image in &self.linear_scene_images {
                state.device.destroy_image(*image, None);
            }
            for allocation in self.linear_scene_image_allocations.drain(..) {
                state
                    .allocator()
                    .free(allocation)
                    .expect("Failed to free linear scene image allocation");
            }
            for image_view in &self.after_tone_mapping_image_views {
                state.device.destroy_image_view(*image_view, None);
            }
            for image in &self.after_tone_mapping_images {
                state.device.destroy_image(*image, None);
            }
            for allocation in self.after_tone_mapping_image_allocations.drain(..) {
                state
                    .allocator()
                    .free(allocation)
                    .expect("Failed to free after tone mapping image allocation");
            }
        }
    }
}
