use anyhow::Result;
use exr::prelude::*;
use half::f16;
use std::path::Path;

pub fn save_mip_image(data: &[u8], width: u32, save_dir: impl AsRef<Path>) -> Result<()> {
    let mut current_mip_width = width;
    let mut data_offset = 0;

    // create directory if it doesn't exist
    let save_dir = save_dir.as_ref();
    if !save_dir.exists() {
        std::fs::create_dir_all(save_dir)?;
    }

    let mut current_mip_level = 0;
    while current_mip_width > 0 {
        let pixel_count = (current_mip_width * current_mip_width) as usize;
        let mip_size = pixel_count * 8 * std::mem::size_of::<f32>();

        if data_offset + mip_size > data.len() {
            anyhow::bail!("Data size is smaller than expected for the mip level");
        }

        let mip_data = &data[data_offset..data_offset + mip_size];

        // u8 -> f16 RGBA
        let f16_pixels: (Vec<[f16; 4]>, Vec<[f16; 4]>) = bytemuck::cast_slice(mip_data)
            .chunks_exact(8)
            .map(|px| {
                (
                    [
                        f16::from_f32(px[0]),
                        f16::from_f32(px[1]),
                        f16::from_f32(px[2]),
                        f16::from_f32(px[3]),
                    ],
                    [
                        f16::from_f32(px[4]),
                        f16::from_f32(px[5]),
                        f16::from_f32(px[6]),
                        f16::from_f32(px[7]),
                    ],
                )
            })
            .collect();

        let mip_path = save_dir.join(format!("latent-texture-0.mip{}.exr", current_mip_level));

        let width = current_mip_width as usize;
        let height = current_mip_width as usize;
        let rgba_pixels = &f16_pixels.0;

        write_rgba_file(mip_path.to_str().unwrap(), width, height, |x, y| {
            let idx = y * width + x;
            let px = rgba_pixels[idx];
            (px[0], px[1], px[2], px[3])
        })?;

        let mip_path = save_dir.join(format!("latent-texture-1.mip{}.exr", current_mip_level));

        let width = current_mip_width as usize;
        let height = current_mip_width as usize;
        let rgba_pixels = &f16_pixels.1;

        write_rgba_file(mip_path.to_str().unwrap(), width, height, |x, y| {
            let idx = y * width + x;
            let px = rgba_pixels[idx];
            (px[0], px[1], px[2], px[3])
        })?;

        current_mip_width /= 2;
        current_mip_level += 1;
        data_offset += mip_size;
    }

    Ok(())
}
