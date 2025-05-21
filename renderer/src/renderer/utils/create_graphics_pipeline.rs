use std::ffi::CString;

use anyhow::Result;
use ash::vk;

use crate::renderer::{texture_manager::TextureManager, vertex::Vertex, vulkan_state::VulkanState};

pub fn create_graphics_pipeline(
    state: &mut VulkanState,
    texture_manager: &mut TextureManager,
    vertex_shader: &[u8],
    fragment_shader: &[u8],
    push_constant_ranges: &[vk::PushConstantRange],
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    // Create shader stage create infos
    let vertex_shader_module = {
        let code = vertex_shader;
        let mut code = std::io::Cursor::new(code);
        let code = ash::util::read_spv(&mut code)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        unsafe { state.device.create_shader_module(&create_info, None)? }
    };
    let fragment_shader_module = {
        let code = fragment_shader;
        let mut code = std::io::Cursor::new(code);
        let code = ash::util::read_spv(&mut code)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        unsafe { state.device.create_shader_module(&create_info, None)? }
    };
    let main_function_name = CString::new("main")?;
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&main_function_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&main_function_name),
    ];

    // Create vertex input state create info
    let binding_description = Vertex::get_binding_descriptions();
    let attribute_descriptions = Vertex::get_attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_descriptions);

    // Create input assembly state info
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Dynamic state create info
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // Create viewport state create info
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    // Create rasterization state create info
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    // Create multisample state create info
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Create color blend attachment states
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];

    // Create color blend state create info
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&color_blend_attachment_states);

    // Create depth stencil state create info
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0);

    // Create pipeline rendering create info
    let rendering_formats = [vk::Format::R8G8B8A8_UNORM];
    let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&rendering_formats)
        .depth_attachment_format(vk::Format::D24_UNORM_S8_UINT);

    // Create pipeline layout
    let set_layouts = texture_manager.descriptor_set_layout();
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(push_constant_ranges);
    let pipeline_layout = unsafe {
        state
            .device
            .create_pipeline_layout(&pipeline_layout_create_info, None)?
    };

    // Create graphics pipeline create info
    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .push_next(&mut pipeline_rendering)
        .layout(pipeline_layout);
    let pipeline = unsafe {
        state
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphics_pipeline_create_info],
                None,
            )
            .expect("Failed to create graphics pipeline")
    }[0];

    // Destroy shader modules
    unsafe {
        state
            .device
            .destroy_shader_module(vertex_shader_module, None);
        state
            .device
            .destroy_shader_module(fragment_shader_module, None);
    }

    Ok((pipeline_layout, pipeline))
}
