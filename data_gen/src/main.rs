use anyhow::Result;
use clap::Parser;

mod data_gen;
mod utils;
mod vulkan_state;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    material: String,
    #[arg(short, long, default_value_t = 1024)]
    texture_size: u32,
    #[arg(short, long, default_value_t = 65536)]
    batch_size: u64,
    #[arg(long, default_value_t = 100)]
    first_phase_shard_size: u64,
    #[arg(long, default_value_t = 300)]
    first_phase_shard_count: u64,
    #[arg(long, default_value_t = 10)]
    second_phase_shard_size: u64,
    #[arg(long, default_value_t = 300)]
    second_phase_shard_count: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Material: {}", args.material);

    // Initialize Vulkan state
    let mut vulkan_state = vulkan_state::VulkanState::new().unwrap();

    match args.material.as_str() {
        "disney-rtnam" => {
            let output_dir = "train/data/disney-rtnam/";
            data_gen::disney_rtnam::data_gen(
                &mut vulkan_state,
                args.texture_size,
                args.batch_size,
                args.first_phase_shard_size,
                args.first_phase_shard_count / 15,
                args.first_phase_shard_count,
                args.second_phase_shard_size,
                args.second_phase_shard_count,
                output_dir,
            )?;
        }
        "pbr-simple" => {
            let base_color_texture_path = "assets/pbr-simple/plane/BaseColor.png";
            let metallic_texture_path = "assets/pbr-simple/plane/Metallic.png";
            let roughness_texture_path = "assets/pbr-simple/plane/Roughness.png";
            let normal_texture_path = "assets/pbr-simple/plane/Normal.png";
            let output_dir = "train/data/pbr-simple/";
            data_gen::pbr_simple::data_gen(
                &mut vulkan_state,
                base_color_texture_path,
                metallic_texture_path,
                roughness_texture_path,
                normal_texture_path,
                args.texture_size,
                args.batch_size,
                args.first_phase_shard_size,
                0,
                args.first_phase_shard_count,
                args.second_phase_shard_size,
                args.second_phase_shard_count,
                output_dir,
            )?;
        }
        _ => {
            println!("Unknown scene: {}", args.material);
        }
    }

    Ok(())
}
