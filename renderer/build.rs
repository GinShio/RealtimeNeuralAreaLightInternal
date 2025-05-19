use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn watch_all_slang_files(dir: &Path) {
    for entry in walkdir::WalkDir::new(dir) {
        let entry = entry.unwrap();
        if entry.path().extension().and_then(|e| e.to_str()) == Some("slang") {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
}

fn compile_slang_shader(src: &Path, dst: &Path, stage: &str, entry_function: &str) {
    assert!(src.exists(), "Source file does not exist: {:?}", src);
    let status = Command::new("slangc")
        .args([
            src.to_str().unwrap(),
            "-target",
            "spirv",
            "-profile",
            "spirv_1_6",
            "-entry",
            entry_function,
            "-stage",
            stage,
            "-o",
            dst.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|_| panic!("Failed to run slangc for {:?}", src));

    if !status.success() {
        panic!("Slang compilation failed for {:?}", src);
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders");
    fs::create_dir_all(&out_dir).unwrap();

    let shader_dir = PathBuf::from("shaders");
    let entry_shaders = [
        (
            "scene/triangle.slang",
            "scene/triangle.vert.spv",
            "vertex",
            "vsMain",
        ),
        (
            "scene/triangle.slang",
            "scene/triangle.frag.spv",
            "fragment",
            "fsMain",
        ),
        (
            "scene/damaged_helmet.slang",
            "scene/damaged_helmet.vert.spv",
            "vertex",
            "vsMain",
        ),
        (
            "scene/damaged_helmet.slang",
            "scene/damaged_helmet.frag.spv",
            "fragment",
            "fsMain",
        ),
        (
            "tone_mapping.slang",
            "tone_mapping.comp.spv",
            "compute",
            "main",
        ),
        (
            "copy_to_swapchain.slang",
            "copy_to_swapchain.vert.spv",
            "vertex",
            "vsMain",
        ),
        (
            "copy_to_swapchain.slang",
            "copy_to_swapchain.frag.spv",
            "fragment",
            "fsMain",
        ),
    ];

    // Watch all .slang files in the shader directory
    watch_all_slang_files(&shader_dir);

    // Compile all shaders
    for (entry_file, output, stage, entry_function) in &entry_shaders {
        let src_path = shader_dir.join(entry_file);
        let out_path = out_dir.join(output);
        let out_dir = out_path.parent().unwrap();
        fs::create_dir_all(out_dir).unwrap();
        compile_slang_shader(&src_path, &out_path, stage, entry_function);
    }
}
