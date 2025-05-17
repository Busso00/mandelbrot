use ocl::{ProQue};
use egui::ColorImage;
use eframe::{egui, App};
use rayon::prelude::*;
//openCL is not faster than rayon
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
    int index = y * width + x;

    double fx = (double)x / (double)width - 0.5;
    double fy = (double)y / (double)height - 0.5;

    double re = center_x + fx * scale;
    double im = center_y + fy * scale;

    double zr = 0.0;
    double zi = 0.0;

    int iter = 0;
    while (iter < max_iter) {
        double zr2 = zr * zr;
        double zi2 = zi * zi;

        double zr_new = zr2 - zi2 + re;
        double zi_new = 2.0 * zr * zi + im;

        zr = zr_new;
        zi = zi_new;

        if (zr * zr + zi * zi > 4.0)
            break;

        iter++;
    }

    output[index] = (double)iter;
}


"#;


pub struct MandelbrotApp {
    center: (f64, f64),
    zoom: f64,
    texture: Option<egui::TextureHandle>,
    max_iterations: u32,
}


impl Default for MandelbrotApp {
    fn default() -> Self {
        Self {
            center: (-0.5, 0.0),
            zoom: 1.0,
            texture: None,
            max_iterations: 2048,
        }
    }
}

fn mandelbrot(c_re: f64, c_im: f64, max_iter: u32) -> u32 {
    let (mut z_re, mut z_im, mut iter) = (0.0, 0.0, 0);
    let (mut z_re2, mut z_im2) = (0.0, 0.0);
    while z_re2 + z_im2 <= 4.0 && iter < max_iter {
        let new_re = z_re2 - z_im2 + c_re;
        let new_im = 2.0 * z_re * z_im + c_im;
        z_re = new_re;
        z_im = new_im;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
        iter += 1;
    }
    iter
}

fn get_color(iter: u32, max_iter: u32) -> [u8; 3] {
    if iter == max_iter {
        return [0, 0, 0];
    }
    // Convert to f64 for consistent floating point precision
    let t = iter as f64 / max_iter as f64;
    let r = if t > 0.5 { ((t - 0.5) * 2.0 * 255.0) as u8 } else { 0 };
    let g = if t < 0.5 {
        (t * 2.0 * 255.0) as u8
    } else {
        ((1.0 - t) * 2.0 * 255.0) as u8
    };
    let b = if t < 0.5 { ((1.0 - t * 2.0) * 255.0) as u8 } else { 0 };
    [r, g, b]
}


impl MandelbrotApp {
    
fn render_mandelbrot_opencl(&self, size: [usize; 2]) -> ColorImage {
        let (width, height) = (size[0], size[1]);
        let scale = 4.0 / self.zoom;
        let max_iter = self.max_iterations as i32;
        let center_x = self.center.0;
        let center_y = self.center.1;

        // Build OpenCL program and queue
        let pro_que = ProQue::builder()
            .src(MANDELBROT_KERNEL)
            .dims([width, height])
            .build()
            .expect("Failed to build OpenCL program");

        // Create output buffer
        let buffer = pro_que.create_buffer::<f64>().expect("Failed to create buffer");

        // Create kernel with arguments
        let kernel = pro_que.kernel_builder("mandelbrot_dd")
            .arg(&buffer)
            .arg(width as i32)
            .arg(height as i32)
            .arg(center_x)
            .arg(center_y)
            .arg(scale)
            .arg(max_iter)
            .build()
            .expect("Failed to build kernel");

        // Enqueue kernel execution
        unsafe { kernel.enq().expect("Failed to enqueue kernel"); }

        // Read results back to CPU
        let mut results = vec![0f64; width * height];
        buffer.read(&mut results).enq().expect("Failed to read buffer");

        // Convert results to pixels using your color map
        let pixels: Vec<egui::Color32> = results
            .iter()
            .map(|&val| {
                let iter = val as u32;
                let color = get_color(iter, max_iter as u32);
                egui::Color32::from_rgb(color[0], color[1], color[2])
            })
            .collect();

        ColorImage {
            size: [width, height],
            pixels,
        }
    }


    fn render_mandelbrot(&self, size: [usize; 2]) -> ColorImage {
        let (w, h) = (size[0], size[1]);
        let scale = 4.0 / self.zoom;
        let max_iter = self.max_iterations;

        // Pre-allocate the pixels vector
        let mut pixels = vec![egui::Color32::BLACK; w * h];

        // Parallel iteration over each pixel
        pixels.par_iter_mut().enumerate().for_each(|(i, pixel)| {
            let x = i % w;
            let y = i / w;

            let re = self.center.0 + (x as f64 / w as f64 - 0.5) * scale;
            let im = self.center.1 + (y as f64 / h as f64 - 0.5) * scale;

            let iter = mandelbrot(re, im, max_iter);
            let [r, g, b] = get_color(iter, max_iter);

            *pixel = egui::Color32::from_rgb(r, g, b);
        });

        ColorImage {
            size: [w, h],
            pixels,
        }
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
                ui.add(egui::Slider::new(&mut self.max_iterations, 128..=8192).logarithmic(true));
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
                let img = self.render_mandelbrot_opencl(size);
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
                let scroll_delta_y = scroll_delta.y;;//scroll x is not detected by OS          
                let drag = ctx.input(|i| i.pointer.delta());
                let drag_x = drag.x;//dreg x is not detected by OS  
                let drag_y = drag.y;//dreg y is not detected by OS 

                if scroll_delta_y != 0.0 {
                    // Get cursor position for zooming toward cursor
                    if let Some(pointer_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                        if is_hovered {
                            println!("scroll {}, {}", scroll_delta_x, scroll_delta_y);
                            
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
                    println!("drag {}, {}", drag_x, drag_y);

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
        Box::new(|_cc| Box::new(MandelbrotApp::default())),
    )
}