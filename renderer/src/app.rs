use std::time::Instant;

use anyhow::Result;
use imgui::Context;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event::{DeviceEvent, DeviceId, Event, StartCause},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::renderer::Renderer;

/// A struct that implements ApplicationHandler.
pub struct App {
    renderer: Option<Renderer>,
    window: Option<Window>,
    imgui: Context,
    platform: WinitPlatform,
    window_size: usize,
    latest_frame: Instant,
}
impl App {
    /// Creates a new instance of the App struct.
    pub fn new() -> Result<Self> {
        let mut imgui = Context::create();
        let platform = WinitPlatform::new(&mut imgui);
        Ok(Self {
            window: None,
            renderer: None,
            imgui,
            platform,
            window_size: 1,
            latest_frame: Instant::now(),
        })
    }

    fn window_size(size: usize) -> PhysicalSize<u32> {
        match size {
            0 => PhysicalSize::new(800, 600),
            1 => PhysicalSize::new(1280, 720),
            2 => PhysicalSize::new(1920, 1080),
            _ => PhysicalSize::new(1280, 720), // Default size
        }
    }
}
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attr = Window::default_attributes()
            .with_title("Vulkan: Test")
            .with_resizable(false)
            .with_inner_size(Self::window_size(self.window_size));
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
        mut event: WindowEvent,
    ) {
        match &mut event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => {
                // Cancel resizing
                let size = Self::window_size(self.window_size);
                if let Some(renderer) = &mut self.renderer {
                    renderer
                        .resize(size.width, size.height)
                        .expect("Failed to resize");
                }
                if let Some(window) = &mut self.window {
                    let _ = window.request_inner_size(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    let prev_window_size = self.window_size;

                    // Generate imgui
                    self.platform
                        .prepare_frame(self.imgui.io_mut(), self.window.as_ref().unwrap())
                        .expect("Failed to prepare frame");
                    let ui = self.imgui.frame();
                    renderer.ui(
                        ui,
                        self.platform.hidpi_factor() as f32,
                        &mut self.window_size,
                    );
                    self.platform
                        .prepare_render(ui, self.window.as_ref().unwrap());
                    let imgui_draw_data = self.imgui.render();

                    renderer.render(imgui_draw_data).expect("Failed to render");

                    if let Some(window) = &mut self.window {
                        // Set window size
                        if prev_window_size != self.window_size {
                            let size = Self::window_size(self.window_size);
                            renderer
                                .resize(size.width, size.height)
                                .expect("Failed to resize");
                            window.request_inner_size(size).unwrap();
                            self.platform.attach_window(
                                self.imgui.io_mut(),
                                window,
                                HiDpiMode::Default,
                            );
                        }

                        // render
                        window.request_redraw();
                    }
                }
            }
            _ => (),
        }

        // imgui handle event
        let event = Event::<()>::WindowEvent { window_id, event };
        self.platform
            .handle_event(self.imgui.io_mut(), self.window.as_ref().unwrap(), &event);
    }
}
