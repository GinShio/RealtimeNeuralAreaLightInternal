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
            "-capability",
            "SPV_NV_cooperative_vector",
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
            "utils/generate_mipmap.slang",
            "utils/generate_mipmap.comp.spv",
            "compute",
            "main",
        ),
        (
            "disney-rtnam/data-gen-1st.slang",
            "disney-rtnam/data-gen-1st.comp.spv",
            "compute",
            "main",
        ),
        (
            "disney-rtnam/data-gen-2nd-material.slang",
            "disney-rtnam/data-gen-2nd-material.comp.spv",
            "compute",
            "main",
        ),
        (
            "disney-rtnam/data-gen-2nd.slang",
            "disney-rtnam/data-gen-2nd.comp.spv",
            "compute",
            "main",
        ),
        (
            "pbr-simple/data-gen-1st.slang",
            "pbr-simple/data-gen-1st.comp.spv",
            "compute",
            "main",
        ),
        (
            "pbr-simple/data-gen-2nd-material.slang",
            "pbr-simple/data-gen-2nd-material.comp.spv",
            "compute",
            "main",
        ),
        (
            "pbr-simple/data-gen-2nd.slang",
            "pbr-simple/data-gen-2nd.comp.spv",
            "compute",
            "main",
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
