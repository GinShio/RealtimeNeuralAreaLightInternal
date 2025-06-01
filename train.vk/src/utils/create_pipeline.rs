use anyhow::Result;
use ash::vk;

use crate::vulkan_state::VulkanState;

pub fn create_compute_pipeline(
    state: &mut VulkanState,
    shader_code: &[u8],
    descriptor_set_layouts: &[vk::DescriptorSetLayout],
    push_constant_ranges: &[vk::PushConstantRange],
) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    let shader_module = {
        let code = shader_code;
        let mut code = std::io::Cursor::new(code);
        let code = ash::util::read_spv(&mut code)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        unsafe { state.device.create_shader_module(&create_info, None)? }
    };

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(descriptor_set_layouts)
        .push_constant_ranges(push_constant_ranges);
    let pipeline_layout = unsafe {
        state
            .device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .unwrap()
    };

    let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()),
        )
        .layout(pipeline_layout);
    let pipeline = unsafe {
        state
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .unwrap()
    }[0];

    // Clean up the shader module
    unsafe {
        state.device.destroy_shader_module(shader_module, None);
    }

    Ok((pipeline, pipeline_layout))
}
