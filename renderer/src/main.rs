use std::ffi::{CStr, CString, c_void};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "validation-enabled")]
use ash::ext::debug_utils;
use ash::{
    Device, Entry, Instance,
    khr::{surface, swapchain},
    vk,
};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use imgui::{Condition, Context, DrawData, Ui};
use imgui_rs_vulkan_renderer::{
    DynamicRendering as ImguiDynamicRendering, Options as ImguiOptions, Renderer as ImguiRenderer,
};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event::{DeviceEvent, DeviceId, Event, StartCause},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

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

struct Swapchain {
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}
impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            // position
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: std::mem::offset_of!(Self, pos) as u32,
            },
            // color
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: std::mem::offset_of!(Self, color) as u32,
            },
        ]
    }
}

struct Renderer {
    #[allow(dead_code)]
    entry: Entry,

    instance: Instance,
    #[cfg(feature = "validation-enabled")]
    debug_fn: debug_utils::Instance,
    surface_fn: surface::Instance,

    #[cfg(feature = "validation-enabled")]
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,

    device: Device,
    swapchain_fn: swapchain::Device,

    command_pool: vk::CommandPool,

    allocator: Option<Arc<Mutex<Allocator>>>,

    queue: vk::Queue,
    swapchain: Swapchain,

    scene_images: Vec<vk::Image>,
    scene_image_views: Vec<vk::ImageView>,
    scene_image_allocations: Vec<Allocation>,

    final_images: Vec<vk::Image>,
    final_image_views: Vec<vk::ImageView>,
    final_image_allocations: Vec<Allocation>,

    graphics_pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    #[allow(dead_code)]
    final_pass_descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    final_pass_descriptor_pool: vk::DescriptorPool,
    final_pass_descriptor_sets: Vec<vk::DescriptorSet>,
    final_pass_pipeline_layout: vk::PipelineLayout,
    final_pass_pipeline: vk::Pipeline,

    present_pass_sampler: vk::Sampler,
    present_pass_descriptor_set_layout: vk::DescriptorSetLayout,
    present_pass_descriptor_pool: vk::DescriptorPool,
    present_pass_descriptor_sets: Vec<vk::DescriptorSet>,
    present_pass_pipeline_layout: vk::PipelineLayout,
    present_pass_pipeline: vk::Pipeline,

    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,

    command_buffers: Vec<vk::CommandBuffer>,

    acquire_next_image_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,

    swapchain_suboptimal: bool,

    current_frame_index: u64,

    // imgui
    imgui_renderer: Option<ImguiRenderer>,
    notify_text: &'static str,
}
impl Renderer {
    const MAX_FRAMES_IN_FLIGHT: usize = 3;

    fn new(window: &Window, imgui: &mut Context) -> Result<Self> {
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
            let app_name = CString::new("Slang test")?;
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
            let required_layers = vec![
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

                if let Some(index) = family_index {
                    Some((device, index))
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
            let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true);
            let mut enabled_features =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan_13_features);

            let enabled_extension_names = [vk::KHR_SWAPCHAIN_NAME.as_ptr()];

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
                .min_image_count(Self::MAX_FRAMES_IN_FLIGHT as u32)
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

        // Create command pool
        let command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index as u32)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { device.create_command_pool(&create_info, None)? }
        };

        // Create gpu_allocator
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

        // create scene images
        let (scene_images, scene_image_views, scene_image_allocations) = (0
            ..Self::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                // Create scene image
                let image_create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .mip_levels(1)
                    .array_layers(1)
                    .extent(vk::Extent3D {
                        width: swapchain.extent.width,
                        height: swapchain.extent.height,
                        depth: 1,
                    })
                    .samples(vk::SampleCountFlags::TYPE_1);
                let image = unsafe { device.create_image(&image_create_info, None) }.unwrap();

                // Allocate memory for the image
                let image_requirements = unsafe { device.get_image_memory_requirements(image) };
                let image_allocation = allocator
                    .allocate(&AllocationCreateDesc {
                        name: "scene image",
                        requirements: image_requirements,
                        location: MemoryLocation::GpuOnly,
                        linear: false,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();
                unsafe {
                    device.bind_image_memory(
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
                    device
                        .create_image_view(&image_view_create_info, None)
                        .expect("Failed to create image view")
                };

                // Create a command buffer
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let command_buffer = unsafe {
                    device
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
                    device
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
                        .unwrap();
                    device.cmd_pipeline_barrier2(
                        command_buffer,
                        &vk::DependencyInfo::default()
                            .image_memory_barriers(&image_memory_barriers),
                    );
                    device.end_command_buffer(command_buffer).unwrap();
                }

                // Create a fence
                let fence_create_info = vk::FenceCreateInfo::default();
                let fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };

                // Submit the command buffer
                let buffers_for_submission = [command_buffer];
                let submit_info =
                    vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
                unsafe {
                    device.queue_submit(queue, &[submit_info], fence).unwrap();
                    device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                }

                // Destroy the fence and command buffer
                unsafe {
                    device.destroy_fence(fence, None);
                    device.free_command_buffers(command_pool, &[command_buffer]);
                }

                (image, image_view, image_allocation)
            })
            .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

        // create final images
        let (final_images, final_image_views, final_image_allocations) = (0
            ..Self::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                // Create final image
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
                        width: swapchain.extent.width,
                        height: swapchain.extent.height,
                        depth: 1,
                    })
                    .samples(vk::SampleCountFlags::TYPE_1);
                let image = unsafe { device.create_image(&image_create_info, None) }.unwrap();

                // Allocate memory for the image
                let image_requirements = unsafe { device.get_image_memory_requirements(image) };
                let image_allocation = allocator
                    .allocate(&AllocationCreateDesc {
                        name: "final image",
                        requirements: image_requirements,
                        location: MemoryLocation::GpuOnly,
                        linear: false,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .unwrap();
                unsafe {
                    device.bind_image_memory(
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
                    device
                        .create_image_view(&image_view_create_info, None)
                        .expect("Failed to create image view")
                };

                // Create a command buffer
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let command_buffer = unsafe {
                    device
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
                    device
                        .begin_command_buffer(
                            command_buffer,
                            &vk::CommandBufferBeginInfo::default(),
                        )
                        .unwrap();
                    device.cmd_pipeline_barrier2(
                        command_buffer,
                        &vk::DependencyInfo::default()
                            .image_memory_barriers(&image_memory_barriers),
                    );
                    device.end_command_buffer(command_buffer).unwrap();
                }

                // Create a fence
                let fence_create_info = vk::FenceCreateInfo::default();
                let fence = unsafe { device.create_fence(&fence_create_info, None).unwrap() };

                // Submit the command buffer
                let buffers_for_submission = [command_buffer];
                let submit_info =
                    vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
                unsafe {
                    device.queue_submit(queue, &[submit_info], fence).unwrap();
                    device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
                }

                // Destroy the fence and command buffer
                unsafe {
                    device.destroy_fence(fence, None);
                    device.free_command_buffers(command_pool, &[command_buffer]);
                }

                (image, image_view, image_allocation)
            })
            .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

        // Create graphics pipeline
        let (graphics_pipeline_layout, graphics_pipeline) = {
            // Create shader stage create infos
            let vertex_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vert.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { device.create_shader_module(&create_info, None)? }
            };
            let fragment_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/frag.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { device.create_shader_module(&create_info, None)? }
            };
            let main_function_name = CString::new("main")?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name),
            ];

            // Create vertex input state create info
            let binding_description = Vertex::get_binding_descriptions();
            let attribute_descriptions = Vertex::get_attribute_descriptions();
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_description)
                .vertex_attribute_descriptions(&attribute_descriptions);

            // Create input assembly state info
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            // Dynamic state create info
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            // Create viewport state create info
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            // Create rasterization state create info
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);

            // Create multisample state create info
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // Create color blend attachment states
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)];

            // Create color blend state create info
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(&color_blend_attachment_states);

            // Create pipeline rendering create info
            let rendering_formats = [vk::Format::R8G8B8A8_UNORM];
            let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&rendering_formats);

            // Create pipeline layout
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&[]);
            let pipeline_layout =
                unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

            // Create graphics pipeline create info
            let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .push_next(&mut pipeline_rendering)
                .layout(pipeline_layout);
            let pipeline = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[graphics_pipeline_create_info],
                        None,
                    )
                    .expect("Failed to create graphics pipeline")
            }[0];

            // Destroy shader modules
            unsafe {
                device.destroy_shader_module(vertex_shader_module, None);
                device.destroy_shader_module(fragment_shader_module, None);
            }

            (pipeline_layout, pipeline)
        };

        // Create final pass descriptor set layout
        let final_pass_descriptor_set_layout = {
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let descriptor_set_layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe {
                device
                    .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                    .expect("Failed to create descriptor set layout")
            }
        };
        // Create final pass descriptor pool
        let final_pass_descriptor_pool = {
            let descriptor_pool_size = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1),
            ];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&descriptor_pool_size)
                .max_sets(Self::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe {
                device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)
                    .expect("Failed to create descriptor pool")
            }
        };
        // Create final pass descriptor sets
        let final_pass_descriptor_sets = {
            let mut descriptor_sets = vec![];
            for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
                let set_layouts = [final_pass_descriptor_set_layout];
                let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(final_pass_descriptor_pool)
                    .set_layouts(&set_layouts);
                let descriptor_set = unsafe {
                    device
                        .allocate_descriptor_sets(&descriptor_set_allocate_info)
                        .expect("Failed to allocate descriptor sets")
                };
                descriptor_sets.push(descriptor_set[0]);
            }
            descriptor_sets
        };
        // Update descriptor sets
        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let input_image_info = [vk::DescriptorImageInfo::default()
                .image_view(scene_image_views[i])
                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)];
            let output_image_info = [vk::DescriptorImageInfo::default()
                .image_view(final_image_views[i])
                .image_layout(vk::ImageLayout::GENERAL)];
            let write_descriptor_sets = [
                vk::WriteDescriptorSet::default()
                    .dst_set(final_pass_descriptor_sets[i])
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_binding(0)
                    .image_info(&input_image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(final_pass_descriptor_sets[i])
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .dst_binding(1)
                    .image_info(&output_image_info),
            ];
            unsafe {
                device.update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }
        // Create final pass compute pipeline
        let (final_pass_pipeline_layout, final_pass_pipeline) = {
            // Create shader stage create infos
            let compute_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/final.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { device.create_shader_module(&create_info, None)? }
            };
            let main_function_name = CString::new("main")?;
            let shader_stages = [vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(compute_shader_module)
                .name(&main_function_name)];

            // Create pipeline layout
            let set_layouts = [final_pass_descriptor_set_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let pipeline_layout =
                unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

            // Create compute pipeline create info
            let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(shader_stages[0])
                .layout(pipeline_layout);
            let compute_pipeline = unsafe {
                device
                    .create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &[compute_pipeline_create_info],
                        None,
                    )
                    .expect("Failed to create compute pipeline")
            }[0];

            // Destroy shader modules
            unsafe {
                device.destroy_shader_module(compute_shader_module, None);
            }

            (pipeline_layout, compute_pipeline)
        };

        // Create present pass sampler
        let present_pass_sampler = {
            let sampler_create_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK);
            unsafe {
                device
                    .create_sampler(&sampler_create_info, None)
                    .expect("Failed to create sampler")
            }
        };
        // Create present pass descriptor set layout
        let present_pass_descriptor_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let descriptor_set_layout_create_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            unsafe {
                device
                    .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                    .expect("Failed to create descriptor set layout")
            }
        };
        // Create present pass descriptor pool
        let present_pass_descriptor_pool = {
            let descriptor_pool_size = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)];
            let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&descriptor_pool_size)
                .max_sets(Self::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe {
                device
                    .create_descriptor_pool(&descriptor_pool_create_info, None)
                    .expect("Failed to create descriptor pool")
            }
        };
        // Create present pass descriptor sets
        let present_pass_descriptor_sets = {
            let mut descriptor_sets = vec![];
            for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
                let set_layouts = [present_pass_descriptor_set_layout];
                let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(present_pass_descriptor_pool)
                    .set_layouts(&set_layouts);
                let descriptor_set = unsafe {
                    device
                        .allocate_descriptor_sets(&descriptor_set_allocate_info)
                        .expect("Failed to allocate descriptor sets")
                };
                descriptor_sets.push(descriptor_set[0]);
            }
            descriptor_sets
        };
        // Update descriptor sets
        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let input_image_info = [vk::DescriptorImageInfo::default()
                .image_view(final_image_views[i])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(present_pass_sampler)];
            let write_descriptor_sets = [vk::WriteDescriptorSet::default()
                .dst_set(present_pass_descriptor_sets[i])
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_binding(0)
                .image_info(&input_image_info)];
            unsafe {
                device.update_descriptor_sets(&write_descriptor_sets, &[]);
            }
        }
        // Create present pass pipeline
        let (present_pass_pipeline_layout, present_pass_pipeline) = {
            // Create shader stage create infos
            let vertex_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/present.vert.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { device.create_shader_module(&create_info, None)? }
            };
            let fragment_shader_module = {
                let code = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/present.frag.spv"));
                let mut code = std::io::Cursor::new(code);
                let code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                unsafe { device.create_shader_module(&create_info, None)? }
            };
            let main_function_name = CString::new("main")?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name),
            ];

            // Create vertex input state create info
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

            // Create input assembly state info
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);

            // Dynamic state create info
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            // Create viewport state create info
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            // Create rasterization state create info
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);

            // Create multisample state create info
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            // Create color blend attachment states
            let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)];

            // Create color blend state create info
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .logic_op_enable(false)
                .attachments(&color_blend_attachment_states);

            // Create pipeline rendering create info
            let rendering_formats = [swapchain.format];
            let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&rendering_formats);

            // Create pipeline layout
            let set_layouts = [present_pass_descriptor_set_layout];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let pipeline_layout =
                unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

            // Create graphics pipeline create info
            let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .push_next(&mut pipeline_rendering)
                .layout(pipeline_layout);
            let pipeline = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[graphics_pipeline_create_info],
                        None,
                    )
                    .expect("Failed to create graphics pipeline")
            }[0];

            // Destroy shader modules
            unsafe {
                device.destroy_shader_module(vertex_shader_module, None);
                device.destroy_shader_module(fragment_shader_module, None);
            }

            (pipeline_layout, pipeline)
        };

        // create vertex buffer
        let vertices = vec![
            Vertex {
                pos: [0.0, 0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [-0.5, -0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [0.5, -0.5],
                color: [0.0, 0.0, 1.0],
            },
        ];
        let (vertex_buffer, vertex_buffer_allocation) = {
            let buffer_size = (std::mem::size_of::<Vertex>() * vertices.len()) as u64;

            // create staging buffer
            let staging_buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let staging_buffer =
                unsafe { device.create_buffer(&staging_buffer_create_info, None)? };

            // Allocate memory for the staging buffer
            let staging_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(staging_buffer) };
            let mut staging_buffer_allocation = allocator.allocate(&AllocationCreateDesc {
                name: "vertex staging buffer",
                requirements: staging_buffer_requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            // Bind the staging buffer memory
            unsafe {
                device.bind_buffer_memory(
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
            data.copy_from_slice(bytemuck::cast_slice(&vertices));

            // Create vertex buffer
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let vertex_buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };

            // Allocate memory for the vertex buffer
            let vertex_buffer_requirements =
                unsafe { device.get_buffer_memory_requirements(vertex_buffer) };
            let vertex_buffer_allocation = allocator.allocate(&AllocationCreateDesc {
                name: "vertex buffer",
                requirements: vertex_buffer_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            // Bind the vertex buffer memory
            unsafe {
                device.bind_buffer_memory(
                    vertex_buffer,
                    vertex_buffer_allocation.memory(),
                    vertex_buffer_allocation.offset(),
                )?;
            }

            // Create a command buffer
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer =
                unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? }[0];

            // Record copy command to the command buffer
            unsafe {
                device
                    .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
                device.cmd_copy_buffer2(
                    command_buffer,
                    &vk::CopyBufferInfo2::default()
                        .src_buffer(staging_buffer)
                        .dst_buffer(vertex_buffer)
                        .regions(&[vk::BufferCopy2::default()
                            .src_offset(0)
                            .dst_offset(0)
                            .size(buffer_size)]),
                );
                device.end_command_buffer(command_buffer)?;
            }

            // Create a fence
            let fence_create_info = vk::FenceCreateInfo::default();
            let fence = unsafe { device.create_fence(&fence_create_info, None)? };

            // Submit the command buffer
            let buffers_for_submission = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&buffers_for_submission);
            unsafe {
                device.queue_submit(queue, &[submit_info], fence)?;
                device.wait_for_fences(&[fence], true, u64::MAX)?;
            }

            // Destroy the fence and command buffer
            unsafe {
                device.destroy_fence(fence, None);
                device.free_command_buffers(command_pool, &[command_buffer]);
            }

            // Destroy the staging buffer
            allocator.free(staging_buffer_allocation)?;
            unsafe {
                device.destroy_buffer(staging_buffer, None);
            }

            // Return the vertex buffer and its memory
            (vertex_buffer, vertex_buffer_allocation)
        };

        // Create main command buffers
        let command_buffers = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? }
        };

        // Create synchronization objects
        let (acquire_next_image_semaphores, render_finished_semaphores, fences) = {
            (0..Self::MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    let mut render_finished_semaphore_create_info =
                        vk::SemaphoreTypeCreateInfo::default();
                    let create_info = vk::SemaphoreCreateInfo::default()
                        .push_next(&mut render_finished_semaphore_create_info);
                    let render_finished_semaphore = unsafe {
                        device
                            .create_semaphore(&create_info, None)
                            .expect("Failed to create timeline semaphore")
                    };

                    let acquire_next_image_semaphore_create_info =
                        vk::SemaphoreCreateInfo::default();
                    let acquire_next_image_semaphore = unsafe {
                        device
                            .create_semaphore(&acquire_next_image_semaphore_create_info, None)
                            .expect("Failed to create present semaphore")
                    };

                    let fence_create_info =
                        vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                    let fence = unsafe {
                        device
                            .create_fence(&fence_create_info, None)
                            .expect("Failed to create fence")
                    };

                    (
                        acquire_next_image_semaphore,
                        render_finished_semaphore,
                        fence,
                    )
                })
                .collect::<(Vec<_>, Vec<_>, Vec<_>)>()
        };

        // setup imgui
        let allocator = Arc::new(Mutex::new(allocator));
        let imgui_renderer = ImguiRenderer::with_gpu_allocator(
            allocator.clone(),
            device.clone(),
            queue,
            command_pool,
            ImguiDynamicRendering {
                color_attachment_format: vk::Format::R8G8B8A8_UNORM,
                depth_attachment_format: None,
            },
            imgui,
            Some(ImguiOptions {
                in_flight_frames: Self::MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            }),
        )
        .expect("Failed to create imgui renderer");

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

            allocator: Some(allocator),

            command_pool,

            queue,
            swapchain,

            scene_images,
            scene_image_views,
            scene_image_allocations,

            final_images,
            final_image_views,
            final_image_allocations,

            graphics_pipeline_layout,
            graphics_pipeline,

            final_pass_descriptor_set_layout,
            final_pass_descriptor_pool,
            final_pass_descriptor_sets,
            final_pass_pipeline_layout,
            final_pass_pipeline,

            present_pass_sampler,
            present_pass_descriptor_set_layout,
            present_pass_descriptor_pool,
            present_pass_descriptor_sets,
            present_pass_pipeline_layout,
            present_pass_pipeline,

            vertices,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),

            command_buffers,

            acquire_next_image_semaphores,
            render_finished_semaphores,
            fences,

            swapchain_suboptimal: false,

            current_frame_index: 0,

            imgui_renderer: Some(imgui_renderer),
            notify_text: "",
        })
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;

            for image_view in &self.swapchain.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.swapchain_fn
                .destroy_swapchain(self.swapchain.swapchain, None);
            for semaphore in &self.acquire_next_image_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
        }

        // recreate semaphores
        unsafe {
            self.acquire_next_image_semaphores = (0..Self::MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    let create_info = vk::SemaphoreCreateInfo::default();
                    self.device
                        .create_semaphore(&create_info, None)
                        .expect("Failed to create timeline semaphore")
                })
                .collect();
            self.render_finished_semaphores = (0..Self::MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    let create_info = vk::SemaphoreCreateInfo::default();
                    self.device
                        .create_semaphore(&create_info, None)
                        .expect("Failed to create timeline semaphore")
                })
                .collect();
        }

        // Recreate swapchain
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
                .min_image_count(Self::MAX_FRAMES_IN_FLIGHT as u32)
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

    fn ui(&mut self, ui: &Ui, hidpi_factor: f32) {
        let width = 300.0;
        let height = 200.0;
        let w = ui
            .window("ImGui Color Button Example")
            .size([width, height], Condition::Appearing)
            .position(
                [1280.0 / hidpi_factor - width - 20.0, 20.0],
                Condition::Appearing,
            );
        w.build(|| {
            ui.text_wrapped(
                "Color button is a widget that displays a color value as a clickable rectangle. \
             It also supports a tooltip with detailed information about the color value. \
             Try hovering over and clicking these buttons!",
            );
            ui.text(self.notify_text);

            ui.text("This button is black:");
            if ui.color_button("Black color", [0.0, 0.0, 0.0, 1.0]) {
                self.notify_text = "*** Black button was clicked";
            }

            ui.text("This button is red:");
            if ui.color_button("Red color", [1.0, 0.0, 0.0, 1.0]) {
                self.notify_text = "*** Red button was clicked";
            }

            ui.text("This button is BIG because it has a custom size:");
            if ui
                .color_button_config("Green color", [0.0, 1.0, 0.0, 1.0])
                .size([100.0, 50.0])
                .build()
            {
                self.notify_text = "*** BIG button was clicked";
            }

            ui.text("This button doesn't use the tooltip at all:");
            if ui
                .color_button_config("No tooltip", [0.0, 0.0, 1.0, 1.0])
                .tooltip(false)
                .build()
            {
                self.notify_text = "*** No tooltip button was clicked";
            }
        });
    }

    fn render(&mut self, imgui_draw_data: &DrawData) -> Result<()> {
        if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            return Ok(());
        }

        if self.swapchain_suboptimal {
            self.resize(self.swapchain.extent.width, self.swapchain.extent.height)?;
            self.swapchain_suboptimal = false;
            return Ok(());
        }

        let in_flight_index =
            (self.current_frame_index % Self::MAX_FRAMES_IN_FLIGHT as u64) as usize;

        // Wait and reset fences
        unsafe {
            self.device
                .wait_for_fences(&[self.fences[in_flight_index]], true, u64::MAX)?;
            self.device.reset_fences(&[self.fences[in_flight_index]])?;
        }

        // Acquire next image
        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.swapchain.swapchain)
            .timeout(1_000_000_000)
            .semaphore(self.acquire_next_image_semaphores[in_flight_index])
            .device_mask(1);
        let image_index = match unsafe { self.swapchain_fn.acquire_next_image2(&acquire_info) } {
            Ok((index, sub_optimal)) => {
                if sub_optimal {
                    println!("Swapchain suboptimal");
                    self.swapchain_suboptimal = true;
                }
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("Swapchain out of date");
                self.resize(self.swapchain.extent.width, self.swapchain.extent.height)?;
                return Ok(());
            }
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                println!("Surface lost");
                self.resize(self.swapchain.extent.width, self.swapchain.extent.height)?;
                return Ok(());
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        let image_index = image_index as usize;

        // Begin command buffer
        let command_buffer = self.command_buffers[in_flight_index];
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        // === main pass ===

        // Memory barrier
        // - scene_images[image_index] ReadOnlyOptimal -> ColorAttachmentOptimal
        let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2KHR::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.scene_images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&image_memory_barriers),
            );
        }

        // Begin rendering
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.scene_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.2, 0.3, 1.0],
                },
            })];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            self.device
                .cmd_begin_rendering(command_buffer, &rendering_info);
        }

        // Bind pipeline
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
        }

        // Set viewport and scissor
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(self.swapchain.extent.height as f32)
            .width(self.swapchain.extent.width as f32)
            .height(-(self.swapchain.extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(self.swapchain.extent);
        unsafe {
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        // Bind vertex buffer
        let vertex_buffers = [self.vertex_buffer];
        let offsets = [0];
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
        }

        // Draw
        unsafe {
            self.device
                .cmd_draw(command_buffer, self.vertices.len() as u32, 1, 0, 0);
        }

        // End rendering
        unsafe {
            self.device.cmd_end_rendering(command_buffer);
        }

        // === final pass ===

        // Memory barrier
        // - scene_images[image_index] ColorAttachmentOptimal -> ReadOnlyOptimal
        // - final_images[image_index] ShaderReadOnlyOptimal -> General
        let image_memory_barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
                .image(self.scene_images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.final_images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&image_memory_barriers),
            );
        }

        // bind final pass compute pipeline
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.final_pass_pipeline,
            );
        }

        // bind final pass descriptor sets
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.final_pass_pipeline_layout,
                0,
                &[self.final_pass_descriptor_sets[image_index]],
                &[],
            );
        }

        // Dispatch final pass compute shader
        let x = (self.swapchain.extent.width + 7) / 8;
        let y = (self.swapchain.extent.height + 7) / 8;
        unsafe {
            self.device.cmd_dispatch(command_buffer, x, y, 1);
        }

        // === Dear ImGui pass ===

        // Memory barrier
        // - final_images[image_index] General -> ColorAttachmentOptimal
        let image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2KHR::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.final_images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&image_memory_barriers),
            );
        }

        // Begin rendering
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.final_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            self.device
                .cmd_begin_rendering(command_buffer, &rendering_info);
        }

        // Draw imgui
        self.imgui_renderer
            .as_mut()
            .unwrap()
            .cmd_draw(command_buffer, imgui_draw_data)?;

        // End rendering
        unsafe {
            self.device.cmd_end_rendering(command_buffer);
        }

        // === render final image to swapchain image ===

        // Memory barrier
        // - final_images[image_index] ColorAttachmentOptimal -> ShaderReadOnlyOptimal
        // - swapchain.images[image_index] PresentSrcKHR -> ColorAttachmentOptimal
        let image_memory_barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2KHR::SHADER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.final_images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2KHR::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2KHR::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.swapchain.images[image_index])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&image_memory_barriers),
            );
        }

        // Begin rendering
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlagsKHR::NONE)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            self.device
                .cmd_begin_rendering(command_buffer, &rendering_info);
        }

        // Bind pipeline
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.present_pass_pipeline,
            );
        }

        // Set viewport and scissor
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain.extent.width as f32)
            .height(self.swapchain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(self.swapchain.extent);
        unsafe {
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        }

        // bind descriptor set
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.present_pass_pipeline_layout,
                0,
                &[self.present_pass_descriptor_sets[image_index]],
                &[],
            );
        }

        // Draw
        unsafe {
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }

        // End rendering
        unsafe {
            self.device.cmd_end_rendering(command_buffer);
        }

        // Memory barrier
        // - swapchain.images[image_index] ColorAttachmentOptimal -> PresentSrcKHR
        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2KHR::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2KHR::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.swapchain.images[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&[image_memory_barrier]),
            );
        }

        // End command buffer
        unsafe { self.device.end_command_buffer(command_buffer)? };

        // Submit command buffer
        let command_buffer_infos =
            [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.acquire_next_image_semaphores[in_flight_index])
            .stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)];
        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.render_finished_semaphores[in_flight_index])
            .stage_mask(vk::PipelineStageFlags2KHR::BOTTOM_OF_PIPE)];
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);
        unsafe {
            self.device
                .queue_submit2(self.queue, &[submit_info], self.fences[in_flight_index])?;
        }

        // Present
        let swapchains = [self.swapchain.swapchain];
        let image_indices = [image_index as u32];
        let wait_semaphores = [self.render_finished_semaphores[in_flight_index]];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&wait_semaphores);
        unsafe {
            self.swapchain_fn.queue_present(self.queue, &present_info)?;
        }

        // Update current frame index
        self.current_frame_index =
            (self.current_frame_index + 1) % Self::MAX_FRAMES_IN_FLIGHT as u64;

        Ok(())
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait for device idle");

            let imgui_renderer = self.imgui_renderer.take().unwrap();
            drop(imgui_renderer);

            let mut allocator = Arc::try_unwrap(self.allocator.take().unwrap())
                .unwrap()
                .into_inner()
                .unwrap();

            for image_view in &self.scene_image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            for image in &self.scene_images {
                self.device.destroy_image(*image, None);
            }
            for allocation in self.scene_image_allocations.drain(..) {
                allocator
                    .free(allocation)
                    .expect("Failed to free scene image allocation");
            }
            for image_view in &self.final_image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            for image in &self.final_images {
                self.device.destroy_image(*image, None);
            }
            for allocation in self.final_image_allocations.drain(..) {
                allocator
                    .free(allocation)
                    .expect("Failed to free final image allocation");
            }

            self.device.destroy_sampler(self.present_pass_sampler, None);
            self.device
                .destroy_descriptor_pool(self.present_pass_descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.present_pass_descriptor_set_layout, None);
            self.device
                .destroy_pipeline(self.present_pass_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.present_pass_pipeline_layout, None);

            self.device
                .destroy_descriptor_pool(self.final_pass_descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.final_pass_descriptor_set_layout, None);
            self.device.destroy_pipeline(self.final_pass_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.final_pass_pipeline_layout, None);

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.acquire_next_image_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.fences[i], None);
            }
            allocator
                .free(self.vertex_buffer_allocation.take().unwrap())
                .expect("Failed to free vertex buffer allocation");
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            for image_view in &self.swapchain.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.swapchain_fn
                .destroy_swapchain(self.swapchain.swapchain, None);
            drop(allocator);
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

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    imgui: Context,
    platform: WinitPlatform,
    latest_frame: Instant,
}
impl App {
    fn new() -> Result<Self> {
        let mut imgui = Context::create();
        let platform = WinitPlatform::new(&mut imgui);
        Ok(Self {
            window: None,
            renderer: None,
            imgui,
            platform,
            latest_frame: Instant::now(),
        })
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("Vulkan: Slang test")
            .with_resizable(false)
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = event_loop
            .create_window(attr)
            .expect("Failed to create window");
        self.platform
            .attach_window(self.imgui.io_mut(), &window, HiDpiMode::Default);
        self.renderer =
            Some(Renderer::new(&window, &mut self.imgui).expect("Failed to create renderer"));
        self.window = Some(window);
        self.window.as_ref().unwrap().request_redraw();
        self.latest_frame = Instant::now();
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {
        let now = Instant::now();
        self.imgui
            .io_mut()
            .update_delta_time(now - self.latest_frame);
        self.latest_frame = now;
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.platform
            .prepare_frame(self.imgui.io_mut(), self.window.as_ref().unwrap())
            .expect("Failed to prepare frame");
        self.window.as_ref().unwrap().request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let event = Event::<()>::DeviceEvent { device_id, event };
        self.platform
            .handle_event(self.imgui.io_mut(), self.window.as_ref().unwrap(), &event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer
                        .resize(size.width, size.height)
                        .expect("Failed to resize");
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    // Generate imgui
                    self.platform
                        .prepare_frame(self.imgui.io_mut(), &self.window.as_ref().unwrap())
                        .expect("Failed to prepare frame");
                    let ui = self.imgui.frame();
                    renderer.ui(&ui, self.platform.hidpi_factor() as f32);
                    self.platform
                        .prepare_render(&ui, &self.window.as_ref().unwrap());
                    let imgui_draw_data = self.imgui.render();

                    // render
                    renderer.render(imgui_draw_data).expect("Failed to render");
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            event => {
                let event = Event::<()>::WindowEvent { window_id, event };
                self.platform.handle_event(
                    self.imgui.io_mut(),
                    &self.window.as_ref().unwrap(),
                    &event,
                );
            }
        }
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new()?;
    event_loop.run_app(&mut app)?;
    Ok(())
}
