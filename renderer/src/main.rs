use std::ffi::{CStr, CString, c_void};
use std::mem::ManuallyDrop;

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
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
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

    allocator: ManuallyDrop<Allocator>,

    graphics_queue: vk::Queue,
    swapchain: Swapchain,
    graphics_pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    graphics_command_pool: vk::CommandPool,

    vertices: Vec<Vertex>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<Allocation>,

    command_buffers: Vec<vk::CommandBuffer>,

    acquire_next_image_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,

    swapchain_suboptimal: bool,

    current_frame_index: u64,
}
impl Renderer {
    const MAX_FRAMES_IN_FLIGHT: usize = 3;

    fn new(window: &Window) -> Result<Self> {
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
        let graphics_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

        // Create swapchain
        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain = {
            let format = vk::Format::B8G8R8A8_SRGB;
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

        // Create gpu_allocator
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;

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
            let rendering_formats = [swapchain.format];
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

        // Create command pool
        let graphics_command_pool = {
            let create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index as u32)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            unsafe { device.create_command_pool(&create_info, None)? }
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
                .command_pool(graphics_command_pool)
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
                device.queue_submit(graphics_queue, &[submit_info], fence)?;
                device.wait_for_fences(&[fence], true, u64::MAX)?;
            }

            // Destroy the fence and command buffer
            unsafe {
                device.destroy_fence(fence, None);
                device.free_command_buffers(graphics_command_pool, &[command_buffer]);
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
                .command_pool(graphics_command_pool)
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

            allocator: ManuallyDrop::new(allocator),

            graphics_queue,
            swapchain,
            graphics_pipeline_layout,
            graphics_pipeline,

            graphics_command_pool,

            vertices,
            vertex_buffer,
            vertex_buffer_allocation: Some(vertex_buffer_allocation),

            command_buffers,

            acquire_next_image_semaphores,
            render_finished_semaphores,
            fences,

            swapchain_suboptimal: false,

            current_frame_index: 0,
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
            let format = vk::Format::B8G8R8A8_SRGB;
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

    fn render(&mut self) -> Result<()> {
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

        // Memory barrier PresentSrcKHR -> ColorAttachmentOptimal
        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
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
            });
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfoKHR::default().image_memory_barriers(&[image_memory_barrier]),
            );
        }

        // Begin rendering
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.image_views[image_index])
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

        // Memory barrier ColorAttachmentOptimal -> PresentSrcKHR
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
            self.device.queue_submit2(
                self.graphics_queue,
                &[submit_info],
                self.fences[in_flight_index],
            )?;
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
            self.swapchain_fn
                .queue_present(self.graphics_queue, &present_info)?;
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

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.acquire_next_image_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.fences[i], None);
            }
            self.allocator
                .free(self.vertex_buffer_allocation.take().unwrap())
                .expect("Failed to free vertex buffer allocation");
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            for image_view in &self.swapchain.image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.swapchain_fn
                .destroy_swapchain(self.swapchain.swapchain, None);
            ManuallyDrop::drop(&mut self.allocator);
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
}
impl App {
    fn new() -> Result<Self> {
        Ok(Self {
            window: None,
            renderer: None,
        })
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("Vulkan: Slang test")
            .with_resizable(false);
        let window = event_loop
            .create_window(attr)
            .expect("Failed to create window");
        self.renderer = Some(Renderer::new(&window).expect("Failed to create renderer"));
        self.window = Some(window);
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
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
                    renderer.render().expect("Failed to render");
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
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
