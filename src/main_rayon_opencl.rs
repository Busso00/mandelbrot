use ocl::{ProQue, Buffer};
use egui::ColorImage;
use eframe::{egui, App};
use std::sync::{Arc};

// OpenCL kernel is unchanged
const MANDELBROT_KERNEL: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_dd(
    __global double* output,
    const int width,
    const int height,
    const double center_x,
    const double center_y,
    const double scale,
    const int max_iter
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    int index = y * width + x;

    double fx = (double)x / (double)width - 0.5;
    double fy = (double)y / (double)height - 0.5;

    double re = center_x + fx * scale;
    double im = center_y + fy * scale;
    double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
    int iter = 0;
    while (zr2 + zi2 <= 4.0 && iter < max_iter) {
        zi = 2.0 * zr * zi + im;
        zr = zr2 - zi2 + re;
        zr2 = zr * zr;
        zi2 = zi * zi;
        iter++;
    }

    output[index] = (double)iter;
}
"#;

fn get_color(iter: u32, max_iter: u32) -> [u8; 3] {
    if iter == max_iter { return [0, 0, 0]; }
    let t = (iter % 128) as f64 / 128.0;
    let r = (0.5 + 0.5 * (6.28318 * t).cos()) * 255.0;
    let g = (0.5 + 0.5 * (6.28318 * t + 2.09439).cos()) * 255.0;
    let b = (0.5 + 0.5 * (6.28318 * t + 4.18879).cos()) * 255.0;
    [r as u8, g as u8, b as u8]
}

pub struct MandelbrotApp {
    center: (f64, f64),
    zoom: f64,
    texture: Option<egui::TextureHandle>,
    max_iterations: u32,
    pro_que: Arc<ProQue>,
}

impl MandelbrotApp {
    pub fn new() -> Self {
        let pro_que = ProQue::builder()
            .src(MANDELBROT_KERNEL)
            .build()
            .expect("OpenCL build failed");
        Self {
            center: (-0.5, 0.0),
            zoom: 1.0,
            texture: None,
            max_iterations: 2048,
            pro_que: Arc::new(pro_que),
        }
    }

    /// Renders the *entire* image in one OpenCL dispatch.
    fn render(&self, width: usize, height: usize) -> ColorImage {
        let total_pixels = width * height;
        // 1) Create one big output buffer
        let buffer = Buffer::<f64>::builder()
            .queue(self.pro_que.queue().clone())
            .len(total_pixels)
            .build()
            .expect("Buffer build");
        // 2) Build kernel with all args
        let mut kernel = self.pro_que.kernel_builder("mandelbrot_dd")
            .arg(&buffer)
            .arg(width as i32)
            .arg(height as i32)
            .arg(self.center.0)
            .arg(self.center.1)
            .arg(4.0 / self.zoom)           // scale
            .arg(self.max_iterations as i32)
            .build()
            .expect("Kernel build");
        // 3) Enqueue it over the full 2D range
        unsafe {
            kernel.cmd()
                .global_work_size([width, height])
                .local_work_size([16, 16])   // 16×16 = 256 threads per group
                .enq()
                .expect("Kernel enqueue");
        }
        // 4) Read back all pixels
        let mut raw = vec![0.0f64; total_pixels];
        buffer.read(&mut raw).enq().expect("Read buffer");
        // 5) Convert to egui::Color32
        let pixels = raw.into_iter()
            .map(|v| {
                let it = v as u32;
                let [r,g,b] = get_color(it, self.max_iterations);
                egui::Color32::from_rgb(r,g,b)
            })
            .collect();

        ColorImage { size: [width, height], pixels }
    }
}



impl App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
       
        egui::CentralPanel::default().show(ctx, |ui| {
            let available_size = egui::Vec2::new(512.0, 512.0);
            let size = [available_size.x as usize, available_size.y as usize];
            // Pan with drag (improved)
            
            // UI Controls
            ui.horizontal(|ui| {
                ui.label("Iterations:");
                let old_iter = self.max_iterations;
                ui.add(egui::Slider::new(&mut self.max_iterations, 128..=4096).logarithmic(true));
                if old_iter != self.max_iterations {
                    self.texture = None; // Force redraw if iterations changed
                }
                
                ui.separator();
                
                ui.label(format!("Zoom: {:.2}x", self.zoom));
                ui.label(format!("Center: ({:.6}, {:.6})", self.center.0, self.center.1));
                
                if ui.button("Reset View").clicked() {
                    self.center = (-0.5, 0.0);
                    self.zoom = 1.0;
                    self.texture = None;
                }
            });
            
            // Mandelbrot rendering
            // Redraw image if window size changes or parameters change
            if self.texture.is_none() || self.texture.as_ref().unwrap().size() != size {
                let img = self.render(size[0], size[1]);
                self.texture = Some(ui.ctx().load_texture("mandelbrot", img, Default::default()));
            }

            if let Some(tex) = &self.texture {
                let response = ui.image(tex);
                
                let mut need_redraw = false;

                
                
                // Pan with arrow keys (always available)
                let pan_speed = 0.05 / self.zoom;
                ctx.input(|i| {
                    if i.key_pressed(egui::Key::ArrowLeft) {
                        self.center.0 -= pan_speed;
                        need_redraw = true;
                    }
                    if i.key_pressed(egui::Key::ArrowRight) {
                        self.center.0 += pan_speed;
                        need_redraw = true;
                    }
                    if i.key_pressed(egui::Key::ArrowUp) {
                        self.center.1 -= pan_speed;
                        need_redraw = true;
                    }
                    if i.key_pressed(egui::Key::ArrowDown) {
                        self.center.1 += pan_speed;
                        need_redraw = true;
                    }
                });
                
                // Zoom controls
                let is_hovered = response.hovered();
                
                // Zoom with scroll wheel (anywhere on image)
                let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
                let scroll_delta_x = scroll_delta.x;
                let scroll_delta_y = scroll_delta.y;//scroll x is not detected by OS          
                let drag = ctx.input(|i| i.pointer.delta());
                let drag_x = drag.x;//dreg x is not detected by OS  
                let drag_y = drag.y;//dreg y is not detected by OS 

                if scroll_delta_y != 0.0 {
                    // Get cursor position for zooming toward cursor
                    if let Some(pointer_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                        if is_hovered {
                            
                                // Calculate position in fractal space before zoom
                                let rel_x = (pointer_pos.x - ui.min_rect().left()) / available_size.x;
                                let rel_y = (pointer_pos.y - ui.min_rect().top()) / available_size.y;
                                let scale = 4.0 / self.zoom;
                                let mouse_re = self.center.0 + (rel_x - 0.5) as f64 * scale;
                                let mouse_im = self.center.1 + (rel_y - 0.5) as f64 * scale;
                                
                                // Apply zoom
                                let zoom_delta = 1.1f64.powf(scroll_delta_y as f64 * 0.1);
                                self.zoom *= zoom_delta;
                                
                                // Adjust center to keep mouse position fixed on the same fractal point
                                let new_scale = 4.0 / self.zoom;
                                self.center.0 = mouse_re - (rel_x - 0.5) as f64 * new_scale;
                                self.center.1 = mouse_im - (rel_y - 0.5) as f64 * new_scale;
                                
                                need_redraw = true;
                            }
                        }
                } else {
                    // Not scrolling

                    if (drag_x != 0.0 || drag_y != 0.0) &&  ctx.input(|i| i.pointer.is_decidedly_dragging()) {
                        let scale = 4.0 / self.zoom;
                        // Convert f32 drag values to f64 for calculation
                        self.center.0 -= drag_x as f64 / size[0] as f64 * scale;
                        self.center.1 -= drag_y as f64 / size[1] as f64 * scale;
                        need_redraw = true;
                    }
                }
                

            
               
                if need_redraw {
                    self.texture = None;
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    // Just use the default options since the exact window size field names have changed
    // between eframe versions
    let mut native_options = eframe::NativeOptions {
        ..Default::default()
    };

    native_options.viewport.inner_size = Some((512.0+16.0, 512.0+40.0).into());

    eframe::run_native(
        "Mandelbrot",
        native_options,
        Box::new(|_cc| Box::new(MandelbrotApp::new())),
    )
}
