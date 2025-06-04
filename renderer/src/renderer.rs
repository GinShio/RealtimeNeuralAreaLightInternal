use std::time::Instant;

use anyhow::Result;
use ash::vk;
use imgui::{Context, DrawData, Ui};
use winit::window::Window;

mod model_data;
mod network;
mod pass;
mod render_images;
mod scene;
mod texture_manager;
mod utils;
mod vertex;
mod vulkan_state;

use vulkan_state::VulkanState;

/// renderer struct.
pub struct Renderer {
    state: VulkanState,

    render_images: render_images::RenderImages,

    texture_manager: texture_manager::TextureManager,

    scenes: Vec<Box<dyn scene::Scene>>,

    scene_pass: pass::ScenePass,
    tone_mapping_pass: pass::ToneMappingPass,
    imgui_pass: pass::ImGuiPass,
    copy_to_swapchain_pass: pass::CopyToSwapchainPass,

    command_buffers: Vec<vk::CommandBuffer>,

    acquire_next_image_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,

    swapchain_suboptimal: bool,
    current_frame_index: u64,

    current_scene_index: usize,

    render_time_counter: u64,
    render_time: [f32; 100],
}
impl Renderer {
    /// Maximum number of frames in flight.
    const MAX_FRAMES_IN_FLIGHT: usize = 1;
    const IMAGE_COUNT: usize = Self::MAX_FRAMES_IN_FLIGHT + 1;

    /// Creates a new instance of the Renderer struct.
    pub fn new(window: &Window, imgui: &mut Context) -> Result<Self> {
        let mut state = VulkanState::new(window)?;

        // Create render images
        let render_images = render_images::RenderImages::new(&mut state)?;

        // Create texture manager
        let mut texture_manager = texture_manager::TextureManager::new(&mut state)?;

        // Create Scenes
        let scenes: Vec<Box<dyn scene::Scene>> = vec![
            scene::TriangleScene::new(&mut state, &mut texture_manager)?,
            scene::DamagedHelmetScene::new(&mut state, &mut texture_manager)?,
            scene::DisneyRtxnsScene::new(&mut state, &mut texture_manager)?,
            scene::DisneyRtnamScene::new(&mut state, &mut texture_manager)?,
            scene::AreaLightScene::new(&mut state, &mut texture_manager)?,
        ];

        // Create pass
        let scene_pass = pass::ScenePass::new();
        let tone_mapping_pass = pass::ToneMappingPass::new(&state, &render_images)?;
        let imgui_pass = pass::ImGuiPass::new(&state, imgui)?;
        let copy_to_swapchain_pass = pass::CopyToSwapchainPass::new(&state, &render_images)?;

        // Create main command buffers
        let command_buffers = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(state.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(Self::MAX_FRAMES_IN_FLIGHT as u32);
            unsafe {
                state
                    .device
                    .allocate_command_buffers(&command_buffer_allocate_info)?
            }
        };

        // Create synchronization objects
        let acquire_next_image_semaphores = (0..Self::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                let create_info = vk::SemaphoreCreateInfo::default();
                unsafe {
                    state
                        .device
                        .create_semaphore(&create_info, None)
                        .expect("Failed to create timeline semaphore")
                }
            })
            .collect::<Vec<_>>();
        let render_finished_semaphores = (0..state.swapchain.images.len())
            .map(|_| {
                let create_info = vk::SemaphoreCreateInfo::default();
                unsafe {
                    state
                        .device
                        .create_semaphore(&create_info, None)
                        .expect("Failed to create timeline semaphore")
                }
            })
            .collect::<Vec<_>>();
        let fences = (0..Self::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                let fence_create_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe {
                    state
                        .device
                        .create_fence(&fence_create_info, None)
                        .expect("Failed to create fence")
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            state,

            render_images,

            texture_manager,

            scenes,

            scene_pass,
            tone_mapping_pass,
            imgui_pass,
            copy_to_swapchain_pass,

            command_buffers,

            acquire_next_image_semaphores,
            render_finished_semaphores,
            fences,

            swapchain_suboptimal: false,
            current_frame_index: 0,

            current_scene_index: 0,

            render_time_counter: 0,
            render_time: [0.0; 100],
        })
    }

    /// Resizes the swapchain and recreates the render images.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.state.device.device_wait_idle()?;

            for image_view in &self.state.swapchain.image_views {
                self.state.device.destroy_image_view(*image_view, None);
            }
            for semaphore in &self.acquire_next_image_semaphores {
                self.state.device.destroy_semaphore(*semaphore, None);
            }
            for semaphore in &self.render_finished_semaphores {
                self.state.device.destroy_semaphore(*semaphore, None);
            }
        }

        // recreate swapchain
        self.state.recreate_swapchain(width, height)?;

        // recreate semaphores
        unsafe {
            self.acquire_next_image_semaphores = (0..Self::MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    let create_info = vk::SemaphoreCreateInfo::default();
                    self.state
                        .device
                        .create_semaphore(&create_info, None)
                        .expect("Failed to create timeline semaphore")
                })
                .collect();
            self.render_finished_semaphores = (0..self.state.swapchain.images.len())
                .map(|_| {
                    let create_info = vk::SemaphoreCreateInfo::default();
                    self.state
                        .device
                        .create_semaphore(&create_info, None)
                        .expect("Failed to create timeline semaphore")
                })
                .collect();
        }

        // recreate render images
        self.render_images.recreate(&mut self.state)?;

        // update descriptor sets
        self.tone_mapping_pass
            .update_render_images(&self.state, &self.render_images);
        self.copy_to_swapchain_pass
            .update_render_images(&self.state, &self.render_images);

        Ok(())
    }

    /// ImGui UI function.
    pub fn ui(&mut self, ui: &Ui, hidpi_factor: f32, window_size: &mut usize) {
        let mut render_time_sum = 0.0;
        for i in 0..(100.min(self.render_time_counter) as usize) {
            render_time_sum += self.render_time[i];
        }
        let render_time = render_time_sum / self.render_time_counter.min(100) as f32;

        self.imgui_pass.ui(
            ui,
            hidpi_factor,
            render_time,
            window_size,
            &mut self.current_scene_index,
            &mut self.scenes,
        );
    }

    /// Main render function.
    pub fn render(&mut self, imgui_draw_data: &DrawData) -> Result<()> {
        let start_time = Instant::now();

        if self.state.swapchain.extent.width == 0 || self.state.swapchain.extent.height == 0 {
            std::thread::sleep(std::time::Duration::from_millis(16));
            return Ok(());
        }

        if self.swapchain_suboptimal {
            self.resize(
                self.state.swapchain.extent.width,
                self.state.swapchain.extent.height,
            )?;
            self.swapchain_suboptimal = false;
            return Ok(());
        }

        let in_flight_index =
            (self.current_frame_index % Self::MAX_FRAMES_IN_FLIGHT as u64) as usize;

        // Wait and reset fences
        unsafe {
            self.state
                .device
                .wait_for_fences(&[self.fences[in_flight_index]], true, u64::MAX)?;
            self.state
                .device
                .reset_fences(&[self.fences[in_flight_index]])?;
        }

        // Acquire next image
        let acquire_info = vk::AcquireNextImageInfoKHR::default()
            .swapchain(self.state.swapchain.swapchain)
            .timeout(1_000_000_000)
            .semaphore(self.acquire_next_image_semaphores[in_flight_index])
            .device_mask(1);
        let image_index =
            match unsafe { self.state.swapchain_fn.acquire_next_image2(&acquire_info) } {
                Ok((index, sub_optimal)) => {
                    if sub_optimal {
                        self.swapchain_suboptimal = true;
                    }
                    index
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    println!("Swapchain out of date");
                    self.resize(
                        self.state.swapchain.extent.width,
                        self.state.swapchain.extent.height,
                    )?;
                    return Ok(());
                }
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => {
                    println!("Surface lost");
                    self.resize(
                        self.state.swapchain.extent.width,
                        self.state.swapchain.extent.height,
                    )?;
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
            self.state
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        // Select scene
        let scene = &mut self.scenes[self.current_scene_index];

        // Record passes
        self.scene_pass.cmd_draw(
            &self.state,
            &self.texture_manager,
            command_buffer,
            image_index,
            &self.render_images,
            scene,
        );
        self.tone_mapping_pass.cmd_draw(
            &self.state,
            command_buffer,
            image_index,
            &self.render_images,
        );
        self.imgui_pass.cmd_draw(
            &self.state,
            command_buffer,
            image_index,
            &self.render_images,
            imgui_draw_data,
        );
        self.copy_to_swapchain_pass.cmd_draw(
            &self.state,
            command_buffer,
            image_index,
            &self.render_images,
        );

        // End command buffer
        unsafe { self.state.device.end_command_buffer(command_buffer)? };

        // Submit command buffer
        let command_buffer_infos =
            [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.acquire_next_image_semaphores[in_flight_index])
            .stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)];
        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.render_finished_semaphores[image_index])
            .stage_mask(vk::PipelineStageFlags2KHR::BOTTOM_OF_PIPE)];
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);
        unsafe {
            self.state.device.queue_submit2(
                self.state.queue,
                &[submit_info],
                self.fences[in_flight_index],
            )?;
        }

        // Present
        let swapchains = [self.state.swapchain.swapchain];
        let image_indices = [image_index as u32];
        let wait_semaphores = [self.render_finished_semaphores[image_index]];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&wait_semaphores);
        unsafe {
            self.state
                .swapchain_fn
                .queue_present(self.state.queue, &present_info)?;
        }

        // Update current frame index
        self.current_frame_index =
            (self.current_frame_index + 1) % Self::MAX_FRAMES_IN_FLIGHT as u64;

        self.render_time[(self.render_time_counter % 100) as usize] =
            start_time.elapsed().as_secs_f32();
        self.render_time_counter += 1;

        Ok(())
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.state
                .device
                .device_wait_idle()
                .expect("Failed to wait for state.device idle");

            self.copy_to_swapchain_pass.destroy(&self.state);
            self.imgui_pass.destroy();
            self.tone_mapping_pass.destroy(&self.state);
            self.scene_pass.destroy();

            for scene in &mut self.scenes {
                scene.destroy(&mut self.state);
            }

            self.texture_manager.destroy(&mut self.state);

            self.render_images.destroy(&mut self.state);

            for semaphore in &self.acquire_next_image_semaphores {
                self.state.device.destroy_semaphore(*semaphore, None);
            }
            for semaphore in &self.render_finished_semaphores {
                self.state.device.destroy_semaphore(*semaphore, None);
            }
            for fence in &self.fences {
                self.state.device.destroy_fence(*fence, None);
            }
        }
    }
}
