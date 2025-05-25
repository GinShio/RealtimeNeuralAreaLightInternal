use anyhow::Result;
use clap::Parser;

mod network;
mod train;
mod utils;
mod vulkan_state;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    scene: String,
    #[arg(short, long, default_value_t = 10000)]
    epochs: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Scene: {}", args.scene);
    println!("Epochs: {}", args.epochs);

    // Initialize Vulkan state
    let mut vulkan_state = vulkan_state::VulkanState::new().unwrap();

    match args.scene.as_str() {
        "disney-rtxns" => {
            train::disney_rtxns::train(&mut vulkan_state, args.epochs)?;
        }
        _ => {
            println!("Unknown scene: {}", args.scene);
        }
    }

    Ok(())
}
