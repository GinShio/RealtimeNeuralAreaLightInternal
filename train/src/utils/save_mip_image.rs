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

    while current_mip_width > 0 {
        let pixel_count = (current_mip_width * current_mip_width) as usize;
        let mip_size = pixel_count * 4 * std::mem::size_of::<f32>();

        if data_offset + mip_size > data.len() {
            anyhow::bail!("Data size is smaller than expected for the mip level");
        }

        let mip_data = &data[data_offset..data_offset + mip_size];

        // u8 -> f16 RGBA
        let f16_pixels: Vec<[f16; 4]> = mip_data
            .chunks_exact(16) // 4ch * 2bytes
            .map(|chunk| {
                [
                    f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
                    f32::from_bits(u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]])),
                    f32::from_bits(u32::from_le_bytes([
                        chunk[8], chunk[9], chunk[10], chunk[11],
                    ])),
                    f32::from_bits(u32::from_le_bytes([
                        chunk[12], chunk[13], chunk[14], chunk[15],
                    ])),
                ]
            })
            .map(|px| {
                [
                    f16::from_f32(px[0]),
                    f16::from_f32(px[1]),
                    f16::from_f32(px[2]),
                    f16::from_f32(px[3]),
                ]
            })
            .collect();

        let mip_path = save_dir.join(format!("latent-texture.{}.exr", current_mip_width));

        let width_usize = current_mip_width as usize;
        let height_usize = current_mip_width as usize;
        let rgba_pixels = &f16_pixels;

        write_rgba_file(
            mip_path.to_str().unwrap(),
            width_usize,
            height_usize,
            |x, y| {
                let idx = y * width_usize + x;
                let px = rgba_pixels[idx];
                (px[0], px[1], px[2], px[3])
            },
        )?;

        current_mip_width /= 2;
        data_offset += mip_size;
    }

    Ok(())
}
