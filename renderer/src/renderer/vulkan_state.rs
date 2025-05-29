use std::ffi::{CStr, CString, c_void};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
#[cfg(feature = "validation-enabled")]
use ash::ext::debug_utils;
use ash::{
    Device, Entry, Instance,
    ext::shader_replicated_composites,
    khr::{surface, swapchain},
    nv::cooperative_vector,
    vk,
};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::renderer::Renderer;

extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

/// A struct that holds the Swapchain state.
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

/// A struct that holds the Vulkan state.
pub struct VulkanState {
    #[allow(dead_code)]
    entry: Entry,

    instance: Instance,
    #[cfg(feature = "validation-enabled")]
    debug_fn: debug_utils::Instance,
    surface_fn: surface::Instance,

    #[cfg(feature = "validation-enabled")]
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,

    surface: vk::SurfaceKHR,

    pub physical_device: vk::PhysicalDevice,

    pub device: Device,
    pub swapchain_fn: swapchain::Device,
    pub cooperative_vector_fn: cooperative_vector::Device,

    pub queue: vk::Queue,
    pub swapchain: Swapchain,

    pub command_pool: vk::CommandPool,

    allocator: Option<Arc<Mutex<Allocator>>>,
}
impl VulkanState {
    /// Creates a new VulkanState instance.
    pub fn new(window: &Window) -> Result<Self> {
        // Load Vulkan library from the system
        let entry = unsafe { Entry::load()? };

        // Debug Utils Messenger Create Info
        let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        // Create Vulkan instance
        let instance = {
            // Application info
            // Use Vulkan 1.3
            let app_name = CString::new("Realtime Neural Area Light")?;
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_3)
                .engine_name(&app_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0));

            // Winit required extensions
            let winit_required_extensions =
                ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?;

            // Additional required extensions
            let required_extensions = [
                #[cfg(feature = "validation-enabled")]
                debug_utils::NAME.as_ptr(),
            ];

            // Enabled extensions
            let enabled_extensions = winit_required_extensions
                .iter()
                .cloned()
                .chain(required_extensions)
                .collect::<Vec<_>>();

            // Required layers
            let required_layers = [
                #[cfg(feature = "validation-enabled")]
                CString::new("VK_LAYER_KHRONOS_validation")?,
            ];
            let enabled_layers = required_layers
                .iter()
                .map(|name| name.as_ptr())
                .collect::<Vec<_>>();

            // Create instance
            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&enabled_extensions)
                .enabled_layer_names(&enabled_layers)
                .push_next(&mut debug_utils_messenger_create_info);
            unsafe { entry.create_instance(&create_info, None)? }
        };

        // Create debug utils messenger
        #[cfg(feature = "validation-enabled")]
        let debug_fn = debug_utils::Instance::new(&entry, &instance);
        #[cfg(feature = "validation-enabled")]
        let debug_utils_messenger = unsafe {
            debug_fn.create_debug_utils_messenger(&debug_utils_messenger_create_info, None)?
        };

        // Create surface
        let surface_fn = surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )?
        };

        // Select physical device
        let (physical_device, queue_family_index) = {
            let physical_devices = unsafe { instance.enumerate_physical_devices()? };
            if physical_devices.is_empty() {
                panic!("No physical devices found");
            }

            // Pick the first physical device that contains a queue family
            // that supports graphics and presentation
            let physical_device = physical_devices.into_iter().find_map(|device| {
                let queue_family_properties =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };

                // Check if the queue family supports graphics and presentation
                let family_index = queue_family_properties
                    .iter()
                    .enumerate()
                    .filter(|(i, family)| {
                        let support_graphics =
                            family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                        let support_present = unsafe {
                            let check_surface_support = surface_fn
                                .get_physical_device_surface_support(device, *i as u32, surface)
                                .unwrap();
                            let check_surface_formats = surface_fn
                                .get_physical_device_surface_formats(device, surface)
                                .map(|formats| !formats.is_empty())
                                .unwrap();
                            let check_present_modes = surface_fn
                                .get_physical_device_surface_present_modes(device, surface)
                                .map(|modes| !modes.is_empty())
                                .unwrap();
                            check_surface_support && check_surface_formats && check_present_modes
                        };
                        support_graphics && support_present
                    })
                    .map(|(i, _)| i)
                    .next();

                // Check if the device supports bindless textures
                let mut descriptor_indexing_features =
                    vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::default();
                let mut features2 = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut descriptor_indexing_features);
                unsafe { instance.get_physical_device_features2(device, &mut features2) };
                let support_bindless_textures = descriptor_indexing_features
                    .shader_sampled_image_array_non_uniform_indexing
                    == vk::TRUE
                    && descriptor_indexing_features
                        .descriptor_binding_sampled_image_update_after_bind
                        == vk::TRUE
                    && descriptor_indexing_features.descriptor_binding_variable_descriptor_count
                        == vk::TRUE
                    && descriptor_indexing_features.descriptor_binding_partially_bound == vk::TRUE
                    && descriptor_indexing_features.runtime_descriptor_array == vk::TRUE;


                    // extension properties
                    let extension_props = unsafe {
                        instance
                            .enumerate_device_extension_properties(device)
                            .unwrap_or_default()
                    };

                // Check if support required extensions
                let support_cooperative_extension = extension_props.iter().any(|ext| {
                    let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                    ext_name == cooperative_vector::NV_COOPERATIVE_VECTOR_NAME
                });
                let support_shader_replicated_extension = extension_props.iter().any(|ext| {
                    let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                    ext_name == shader_replicated_composites::EXT_SHADER_REPLICATED_COMPOSITES_NAME
                });
                let support_extensions = support_cooperative_extension
                    && support_shader_replicated_extension;

                // Check if support required features
                let mut cooperative_features =
                    cooperative_vector::PhysicalDeviceCooperativeVectorFeaturesNV::default();
                    let mut replicated_composites_features =
                        shader_replicated_composites::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT::default();
                let mut features2 =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut cooperative_features).push_next(&mut replicated_composites_features);
                unsafe {
                    instance.get_physical_device_features2(device, &mut features2);
                }
                let support_features = cooperative_features.cooperative_vector == vk::TRUE
                    && replicated_composites_features.shader_replicated_composites == vk::TRUE;

                if support_bindless_textures && support_extensions && support_features {
                    family_index.map(|index| (device, index))
                } else {
                    None
                }
            });

            if let Some((device, index)) = physical_device {
                (device, index)
            } else {
                panic!("No suitable physical device found");
            }
        };

        // Create Device
        let device = {
            // Use Vulkan 1.1 features:
            // - storage_buffer16_bit_access
            // Use Vulkan 1.2 features:
            // - shader_float_16
            // - shader_sampled_image_array_non_uniform_indexing
            // - descriptor_binding_sampled_image_update_after_bind
            // - descriptor_binding_variable_descriptor_count
            // - descriptor_binding_partially_bound
            // - runtime_descriptor_array
            // Use Vulkan 1.3 features:
            // - synchronization2
            // - dynamic rendering
            // - extended dynamic state
            // Use cooperative vector features:
            // - cooperative_vector
            // - shader_replicated_composites
            let mut vulkan_11_features =
                vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
                .shader_float16(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true);
            let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true);
            let mut extended_dynamic_state =
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
                    .extended_dynamic_state(true);
            let mut replicated_composites_features =
                shader_replicated_composites::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT::default()
                    .shader_replicated_composites(true);
            let mut cooperative_vector_features =
                cooperative_vector::PhysicalDeviceCooperativeVectorFeaturesNV::default()
                    .cooperative_vector(true);
            let mut enabled_features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut vulkan_11_features)
                .push_next(&mut vulkan_12_features)
                .push_next(&mut vulkan_13_features)
                .push_next(&mut extended_dynamic_state)
                .push_next(&mut replicated_composites_features)
                .push_next(&mut cooperative_vector_features);

            let enabled_extension_names = [
                vk::KHR_SWAPCHAIN_NAME.as_ptr(),
                cooperative_vector::NV_COOPERATIVE_VECTOR_NAME.as_ptr(),
                shader_replicated_composites::EXT_SHADER_REPLICATED_COMPOSITES_NAME.as_ptr(),
            ];

            let queue_create_infos = vec![
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index as u32)
                    .queue_priorities(&[1.0]),
            ];
            let create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&enabled_extension_names)
                .push_next(&mut enabled_features);

            unsafe { instance.create_device(physical_device, &create_info, None)? }
        };

        // Get queue
        let queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        // Create swapchain
        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain = {
            let format = vk::Format::B8G8R8A8_UNORM;
            let present_mode = vk::PresentModeKHR::FIFO;
            let capabilities = unsafe {
                surface_fn.get_physical_device_surface_capabilities(physical_device, surface)?
            };
            let extent = vk::Extent2D {
                width: window.inner_size().width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: window.inner_size().height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            };

            let create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(Renderer::IMAGE_COUNT as u32)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain = unsafe { swapchain_fn.create_swapchain(&create_info, None)? };
            let swapchain_images = unsafe { swapchain_fn.get_swapchain_images(swapchain) }?;
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let swapchain_image_views = swapchain_images
                .iter()
                .map(|&image| {
                    let create_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(subresource_range);
                    unsafe {
                        device
                            .create_image_view(&create_info, None)
                            .expect("Failed to create image view")
                    }
                })
                .collect::<Vec<_>>();
            Swapchain {
                swapchain,
                format,
                extent,
                images: swapchain_images,
                image_views: swapchain_image_views,
            }
        };

        // Create cooperative vector device
        let cooperative_vector_fn = cooperative_vector::Device::new(&instance, &device);

        // Create command pool
        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index as u32)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { device.create_command_pool(&create_info, None)? }
        };

        // Create gpu_allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            entry,
            instance,
            #[cfg(feature = "validation-enabled")]
            debug_fn,
            surface_fn,
            #[cfg(feature = "validation-enabled")]
            debug_utils_messenger,
            surface,
            physical_device,
            device,
            swapchain_fn,
            cooperative_vector_fn,
            queue,
            swapchain,
            command_pool,
            allocator: Some(Arc::new(Mutex::new(allocator))),
        })
    }

    /// Recreates the swapchain.
    pub fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        // Wait for device to be idle before destroying the swapchain
        unsafe {
            self.device.device_wait_idle()?;
        }

        // Destroy old swapchain
        unsafe {
            self.swapchain_fn
                .destroy_swapchain(self.swapchain.swapchain, None);
        }

        // Recreate state.swapchain
        self.swapchain = {
            let format = vk::Format::B8G8R8A8_UNORM;
            let present_mode = vk::PresentModeKHR::FIFO;
            let capabilities = unsafe {
                self.surface_fn
                    .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
            };
            let extent = vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            };

            let create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(Renderer::IMAGE_COUNT as u32)
                .image_format(format)
                .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain = unsafe { self.swapchain_fn.create_swapchain(&create_info, None)? };
            let swapchain_images = unsafe { self.swapchain_fn.get_swapchain_images(swapchain) }?;
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let swapchain_image_views = swapchain_images
                .iter()
                .map(|&image| {
                    let create_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(subresource_range);
                    unsafe {
                        self.device
                            .create_image_view(&create_info, None)
                            .expect("Failed to create image view")
                    }
                })
                .collect::<Vec<_>>();
            Swapchain {
                swapchain,
                format,
                extent,
                images: swapchain_images,
                image_views: swapchain_image_views,
            }
        };

        Ok(())
    }

    /// Begin a single-use command buffer.
    pub fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info).unwrap() };
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }
        command_buffer
    }

    /// End and submit a single-use command buffer, then free it.
    pub fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            self.device
                .queue_submit(self.queue, &[submit_info], vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();

            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
    }

    /// Get a MutexGuard to the allocator.
    pub fn allocator(&mut self) -> MutexGuard<'_, Allocator> {
        self.allocator
            .as_ref()
            .expect("Allocator not initialized")
            .lock()
            .unwrap()
    }

    /// Clone the allocator.
    pub fn clone_allocator(&self) -> Arc<Mutex<Allocator>> {
        self.allocator
            .as_ref()
            .expect("Allocator not initialized")
            .clone()
    }
}
impl Drop for VulkanState {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait for .device idle");

            let allocator = Arc::try_unwrap(self.allocator.take().unwrap())
                .unwrap()
                .into_inner()
                .unwrap();
            drop(allocator);

            for image_view in &self.swapchain.image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            self.swapchain_fn
                .destroy_swapchain(self.swapchain.swapchain, None);

            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            #[cfg(feature = "validation-enabled")]
            self.debug_fn
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None);
        }
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
