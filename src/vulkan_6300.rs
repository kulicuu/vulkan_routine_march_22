
#![feature(drain_filter)]



use erupt::{
    cstr,
    utils::{self, surface},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    vk::{Device, MemoryMapFlags},
};
use cgmath::{Deg, Rad, Matrix4, Point3, Vector3, Vector4};
use nalgebra_glm as glm;
use std::{
    ffi::{c_void, CStr, CString},
    fs,
    fs::{write, OpenOptions},
    io::prelude::*,
    mem::*,
    os::raw::c_char,
    ptr,
    result::Result,
    result::Result::*,
    string::String,
    sync::Arc,
    thread,
    time,
};

use std::time::{Duration, Instant};
use std::thread::sleep;

use smallvec::SmallVec;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use memoffset::offset_of;
use simple_logger::SimpleLogger;
use winit::{
    dpi::PhysicalSize,
    event::{
        Event, KeyboardInput, WindowEvent,
        ElementState, StartCause, VirtualKeyCode,
        DeviceEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    window::Window
};
use structopt::StructOpt;
const TITLE: &str = "vulkan-routine";
const FRAMES_IN_FLIGHT: usize = 2;
const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");
const SHADER_VERT: &[u8] = include_bytes!("../spv/s_400_.vert.spv");
const SHADER_FRAG: &[u8] = include_bytes!("../spv/s1.frag.spv");
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FrameData {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexV3 {
    pos: [f32; 4],
    color: [f32; 4],
}


#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct PushConstants {
    view: glm::Mat4,
}


#[repr(C)]
#[derive(Clone, Debug, Copy)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}


#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    validation_layers: bool,
}



#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Camera {
    location: glm::Vec3,
    target: glm::Vec3,  // not necessarily normalized, this one to one with the look_at (or look_at_rh) function.
    up: glm::Vec3,  // yaw axis in direction of up
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ControlInput {
    roll: i32,
    pitch: i32,
    yaw: i32,
    skew: i32,
}


// static mut log_300: Vec<String> = vec!();
unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let str_99 = String::from(CStr::from_ptr((*p_callback_data).p_message).to_string_lossy());
    // log_300.push(str_99 );
    eprintln!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );
    vk::FALSE
}


pub unsafe fn vulkan_routine_6300
()
{
    println!("\n6300\n");
    routine_pure_procedural();
}



// create a monolith sort of like we start with app.
// we are basically going to reproduce backup, but change names of stuff, and organize it better,
// in order to be able to turn it into an object.

unsafe fn routine_pure_procedural
()
{
    
    let opt = Opt { validation_layers: true };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .with_resizable(false)
        .with_maximized(true)
        .build(&event_loop)
        .unwrap();
    let entry = Arc::new(EntryLoader::new().unwrap());
    let application_name = CString::new("Vulkan-Routine-6300").unwrap();
    let engine_name = CString::new("Peregrine").unwrap();
    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&application_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));
    let mut instance_extensions = surface::enumerate_required_extensions(&window).unwrap();
    if opt.validation_layers {
        instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    let mut instance_layers = Vec::new();
    if opt.validation_layers {
        instance_layers.push(LAYER_KHRONOS_VALIDATION);
    }
    let device_extensions = vec![
        vk::KHR_SWAPCHAIN_EXTENSION_NAME,
        vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk::KHR_RAY_QUERY_EXTENSION_NAME,
        vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk::KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        vk::KHR_SPIRV_1_4_EXTENSION_NAME,
        vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    ];
    let mut device_layers = Vec::new();
    if opt.validation_layers {
        device_layers.push(LAYER_KHRONOS_VALIDATION);
    }
    let instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);
    let instance = Arc::new(InstanceLoader::new(&entry, &instance_info).unwrap());
    let messenger = if opt.validation_layers {
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
            )
            .pfn_user_callback(Some(debug_callback));

        instance.create_debug_utils_messenger_ext(&messenger_info, None).expect("problem creating debug util.");
    } else {
        Default::default()
    };
    let surface = surface::create_surface(&instance, &window, None).unwrap();


    let (physical_device, queue_family, format, present_mode, device_properties) = create_precursors
    (
        &instance,
        surface.clone(),
    ).unwrap();

    println!("\n\n\nUsing physical device: {:?}\n\n\n", CStr::from_ptr(device_properties.device_name.as_ptr()));


    let queue_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();
    let device_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);
    let device = Arc::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap());
    let queue = device.get_device_queue(queue_family, 0);
    let surface_caps = instance.get_physical_device_surface_capabilities_khr(physical_device, surface).unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }



    let swapchain_image_extent = match surface_caps.current_extent {
        vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        } => {
            let PhysicalSize { width, height } = window.inner_size();
            vk::Extent2D { width, height }
        }
        normal => normal,
    };
    let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(swapchain_image_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());
    let swapchain = device.create_swapchain_khr(&swapchain_info, None).unwrap();
    let swapchain_images = device.get_swapchain_images_khr(swapchain, None).unwrap();
    let swapchain_image_views: Vec<_> = swapchain_images
        .iter()
        .map(|swapchain_image| {
            let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRangeBuilder::new()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            device.create_image_view(&image_view_info, None).unwrap()
        })
        .collect();

    let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = device.create_command_pool(&command_pool_info, None).unwrap();
    
    
    
    
 


    let (mut vertices_terr, mut indices_terr) = load_model().unwrap();

    let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);


    
    let ib = buffer_indices
    (
        &device,
        queue,
        command_pool,
        &mut indices_terr,
    ).unwrap();
    
    
    let vb = buffer_vertices
    (
        &device,
        queue,
        command_pool,
        &mut vertices_terr, 
    ).unwrap();



    let info = vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
        .binding_flags(&[vk::DescriptorBindingFlags::empty()]);




    let samplers = [vk::Sampler::null()];
    let binding = vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .immutable_samplers(&samplers);
    let slice = &[binding];
    let info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty()) 
        .bindings(slice);
    let descriptor_set_layout = device.create_descriptor_set_layout(&info, None).unwrap();

    let ubo_size = ::std::mem::size_of::<UniformBufferObject>();
    let mut uniform_buffers: Vec<vk::Buffer> = vec![];
    let mut uniform_buffers_memories: Vec<vk::DeviceMemory> = vec![];
    let swapchain_image_count = swapchain_images.len();


    for _ in 0..swapchain_image_count {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            &device,
            ubo_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            2,
        );
        uniform_buffers.push(uniform_buffer);
        uniform_buffers_memories.push(uniform_buffer_memory);
    }
    let scalar_22 = 1.5;
    let mut uniform_transform = UniformBufferObject {
        model: 
        // Matrix4::from_translation(Vector3::new(5.0, 5.0, 5.0))
        Matrix4::from_angle_y(Deg(1.0))
            * Matrix4::from_nonuniform_scale(scalar_22, scalar_22, scalar_22),
        view: Matrix4::look_at_rh(
            Point3::new(0.40, 0.40, 0.40),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ),
        proj: {
            let mut proj = cgmath::perspective(
                Deg(45.0),
                swapchain_image_extent.width as f32
                    / swapchain_image_extent.height as f32,
                0.1,
                10.0,
            );
            proj[1][1] = proj[1][1] * -1.0;
            proj
        },
    };



    let scalar_33 = 100000000.0;


    let mut camera = Camera {
        location: glm::vec3(1.0 / scalar_33, 1.0 / scalar_33, 1.0 / scalar_33),
        target: glm::vec3(0.0, 0.0, 0.0),
        up: glm::vec3(0.0, 1.0, 0.0),
    };


    let mut view: glm::Mat4 = glm::look_at::<f32>
    (
        &camera.location,
        &camera.target,
        &camera.up,
    );

    let mut push_constant = PushConstants {
        view: view,
    };

    // let mut push_constant = PushConstants {
    //     view: glm::Matrix4::look_at_rh(
    //         Point3::new(0.80, 0.80, 0.80),
    //         Point3::new(0.0, 0.0, 0.0),
    //         Vector3::new(0.0, 0.0, 1.0),
    // };

    let pool_size = vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(swapchain_image_count as u32);

    let pool_sizes = &[pool_size];
    let set_layouts = &[descriptor_set_layout];

    let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(pool_sizes)
        .max_sets(swapchain_image_count as u32);


    let desc_pool = device.create_descriptor_pool(&pool_info, None).unwrap();

    let d_set_alloc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(set_layouts);

    let d_sets = device.allocate_descriptor_sets(&d_set_alloc_info).expect("failed in alloc DescriptorSet");

    let ubo_size = ::std::mem::size_of::<UniformBufferObject>() as u64;


    for i in 0..swapchain_image_count {
        let d_buf_info = vk::DescriptorBufferInfoBuilder::new()
            .buffer(uniform_buffers[i])
            .offset(0)
            .range(ubo_size);

        let d_buf_infos = [d_buf_info];

        let d_write_builder = vk::WriteDescriptorSetBuilder::new()
            .dst_set(d_sets[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&d_buf_infos);

        let d_write_sets = [d_write_builder];

        device.update_descriptor_sets(&d_write_sets, &[]);
        update_uniform_buffer
        (
            &device, 
            &mut uniform_transform, 
            &mut uniform_buffers_memories, 
            &mut uniform_buffers, 
            i as usize, 
            2.3
        );

    }




    let depth_image_info = vk::ImageCreateInfoBuilder::new()
        .flags(vk::ImageCreateFlags::empty())
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::D32_SFLOAT)
        .extent(vk::Extent3D {
            width: swapchain_image_extent.width,
            height: swapchain_image_extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlagBits::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&[0])
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let depth_image = device.create_image(&depth_image_info, None)
            .expect("Failed to create depth (texture) Image.");   

    let dpth_img_mem_reqs = device.get_image_memory_requirements(depth_image);
    let dpth_img_mem_info = vk::MemoryAllocateInfoBuilder::new()
        .memory_type_index(1)
        .allocation_size(dpth_img_mem_reqs.size);
    let depth_image_memory = device.allocate_memory(&dpth_img_mem_info, None)
        .expect("Failed to alloc mem for depth image.");

    device.bind_image_memory(depth_image, depth_image_memory, 0)
        .expect("Failed to bind depth image memory.");
    
    
    let depth_image_view_info = vk::ImageViewCreateInfoBuilder::new()
        .flags(vk::ImageViewCreateFlags::empty())
        .image(depth_image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::D32_SFLOAT)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let depth_image_view = device.create_image_view(&depth_image_view_info, None)
        .expect("Failed to create image view.");

    let entry_point = CString::new("main").unwrap();
    let vert_decoded = utils::decode_spv(SHADER_VERT).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&vert_decoded);
    let shader_vert = device.create_shader_module(&module_info, None).unwrap();
    let frag_decoded = utils::decode_spv(SHADER_FRAG).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&frag_decoded);
    let shader_frag = device.create_shader_module(&module_info, None).unwrap();
    let shader_stages = vec![
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(shader_vert)
            .name(&entry_point),
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(shader_frag)
            .name(&entry_point),
    ];
    let vertex_buffer_bindings_desc_info = vk::VertexInputBindingDescriptionBuilder::new()
        .binding(0)
        .stride(std::mem::size_of::<VertexV3>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vert_buff_att_desc_info_pos = vk::VertexInputAttributeDescriptionBuilder::new()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(offset_of!(VertexV3, pos) as u32,);
    let vert_buff_att_desc_info_color = vk::VertexInputAttributeDescriptionBuilder::new()
        .location(1)
        .binding(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(offset_of!(VertexV3, color) as u32,);
    let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
        .flags(vk::PipelineVertexInputStateCreateFlags::empty())
        .vertex_binding_descriptions(&[vertex_buffer_bindings_desc_info])
        .vertex_attribute_descriptions(&[vert_buff_att_desc_info_pos, vert_buff_att_desc_info_color])
        .build_dangling();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewports = vec![vk::ViewportBuilder::new()
        .x(0.0)
        .y(0.0)
        .width(swapchain_image_extent.width as f32)
        .height(swapchain_image_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissors = vec![vk::Rect2DBuilder::new()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(swapchain_image_extent)];
    let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
        .depth_clamp_enable(true)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::LINE)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlagBits::_1);
    let color_blend_attachments = vec![vk::PipelineColorBlendAttachmentStateBuilder::new()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)];
    let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let pipeline_stencil_info = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
        .depth_test_enable(false)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .front(vk::StencilOpStateBuilder::new().build())
        .back(vk::StencilOpStateBuilder::new().build());

    let desc_layouts_slc = &[descriptor_set_layout];


    let push_constant_range = vk::PushConstantRangeBuilder::new()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<glm::Mat4>() as u32);
    let slice = [push_constant_range];

    

    let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
        .set_layouts(desc_layouts_slc)
        .push_constant_ranges(&slice);


    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None).unwrap();
    let attachments = vec![
        vk::AttachmentDescriptionBuilder::new()
            .format(format.format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        vk::AttachmentDescriptionBuilder::new()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    ];
    let depth_attach_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_attachment_refs = vec![vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];


    let subpasses = vec![vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attach_ref)];


    let dependencies = vec![vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    let render_pass = device.create_render_pass(&render_pass_info, None).unwrap();

   
    let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .depth_stencil_state(&pipeline_stencil_info)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None).unwrap()[0];
    let swapchain_framebuffers: Vec<_> = swapchain_image_views
        .iter()
        .map(|image_view| {
            let attachments = vec![*image_view, depth_image_view];
            let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_image_extent.width)
                .height(swapchain_image_extent.height)
                .layers(1);
            device.create_framebuffer(&framebuffer_info, None).unwrap()
        })
        .collect();

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let cmd_bufs = device.allocate_command_buffers(&cmd_buf_allocate_info).unwrap();




    let now = Instant::now();






    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
        .collect();
    let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| device.create_semaphore(&semaphore_info, None).unwrap())
        .collect();
    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| device.create_fence(&fence_info, None).unwrap())
        .collect();
    let mut images_in_flight: Vec<_> = swapchain_images.iter().map(|_| vk::Fence::null()).collect();
    let mut frame = 0;




    let mut button_push: [bool; 2] = [false; 2];







    let mut control_input = ControlInput {
        roll: 0,
        pitch: 0,
        yaw: 0,
        skew: 0,
    };
    
    #[allow(clippy::collapsible_match, clippy::single_match)]
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Poll;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(keycode),
                state,
                ..
            }) => match (keycode, state) {

                (VirtualKeyCode::Escape, ElementState::Released) => {
                    *control_flow = ControlFlow::Exit
                },
                (winit::event::VirtualKeyCode::Space, ElementState::Released) => {
                    button_push[frame] = true;
                },
                (winit::event::VirtualKeyCode::Right, ElementState::Pressed) => {
                    control_input.roll += 1;
                },
                (winit::event::VirtualKeyCode::Left, ElementState::Pressed) => {
                    control_input.roll -= 1;
                },
                (winit::event::VirtualKeyCode::Up, ElementState::Pressed) => {
                    control_input.pitch -= 1;
                },
                (winit::event::VirtualKeyCode::Down, ElementState::Pressed) => {
                    control_input.pitch += 1;
                },
                (winit::event::VirtualKeyCode::Semicolon, ElementState::Pressed) => {
                    control_input.yaw -= 1;
                },
                (winit::event::VirtualKeyCode::J, ElementState::Pressed) => {
                    control_input.yaw += 1;
                },
                _ => (),

            },

            _ => (),
        },
        Event::MainEventsCleared => {
            device.wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX).unwrap();
            let image_index = device.acquire_next_image_khr
            (
                swapchain,
                u64::MAX,
                image_available_semaphores[frame],
                vk::Fence::null(),
            ).unwrap();


            let delta_time = now.elapsed().as_secs_f32();



            transform_camera(&mut camera, &mut push_constant.view, &mut control_input);



            // push_constant = update_push_constants(push_constant, delta_time).unwrap();


            // if button_push[frame] {
            //     push_constant = update_push_constants(push_constant, delta_time).unwrap();
    
    
            //     // update_uniform_buffer(&device, &mut uniform_transform, &mut uniform_buffers_memories, &mut uniform_buffers, image_index as usize, 3.2);

            // }
            



            // push_constant.view = 

            // update_uniform_buffer(&device, &mut uniform_transform, &mut uniform_buffers_memories, &mut uniform_buffers, image_index as usize, 3.2);
            // push_constant = update_push_constants(push_constant, delta_time).unwrap();




            // mutate_view_matrix(&mut push_constant.view, &mut control_input);


            button_push[frame] = false;





            let image_in_flight = images_in_flight[image_index as usize];
            if !image_in_flight.is_null() {
                device.wait_for_fences(&[image_in_flight], true, u64::MAX).unwrap();
            }
            images_in_flight[image_index as usize] = in_flight_fences[frame];
            let wait_semaphores = vec![image_available_semaphores[frame]];
            // let command_buffers = vec![cmd_bufs[image_index as usize]];

            let command_buffer = cmd_bufs[image_index as usize];
            let framebuffer = swapchain_framebuffers[image_index as usize];


            record_cb_111
            (
                command_buffer,
                &device,
                render_pass,
                framebuffer,
                swapchain_image_extent,
                pipeline,
                pipeline_layout,
                &indices_terr,
                d_sets.clone(),
                vb,
                ib,
                push_constant,
            );
            let command_buffers = [command_buffer];
            let signal_semaphores = vec![render_finished_semaphores[frame]];
            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            let in_flight_fence = in_flight_fences[frame];
            device.reset_fences(&[in_flight_fence]).unwrap();
            device
                .queue_submit(queue, &[submit_info], in_flight_fence)
                .unwrap();
            let swapchains = vec![swapchain];
            let image_indices = vec![image_index];
            let present_info = vk::PresentInfoKHRBuilder::new()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            device.queue_present_khr(queue, &present_info).unwrap();
            frame = (frame + 1) % FRAMES_IN_FLIGHT;




        }
        // Event::LoopDestroyed => unsafe {
        //     device.device_wait_idle().unwrap();
        //     for &semaphore in image_available_semaphores
        //         .iter()
        //         .chain(render_finished_semaphores.iter())
        //     {
        //         device.destroy_semaphore(semaphore, None);
        //     }
        //     for &fence in &in_flight_fences {
        //         device.destroy_fence(fence, None);
        //     }
        //     device.destroy_command_pool(command_pool, None);
        //     for &framebuffer in &swapchain_framebuffers {
        //         device.destroy_framebuffer(framebuffer, None);
        //     }
        //     device.destroy_pipeline(pipeline, None);
        //     device.destroy_render_pass(render_pass, None);
        //     device.destroy_pipeline_layout(pipeline_layout, None);
        //     device.destroy_shader_module(shader_vert, None);
        //     device.destroy_shader_module(shader_frag, None);
        //     for &image_view in &swapchain_image_views {
        //         device.destroy_image_view(image_view, None);
        //     }
        //     device.destroy_swapchain_khr(swapchain, None);
        //     device.destroy_device(None);
        //     instance.destroy_surface_khr(surface, None);
        //     if !messenger.is_null() {
        //         instance.destroy_debug_utils_messenger_ext(messenger, None);
        //     }
        //     instance.destroy_instance(None);
        //     println!("Exited cleanly");
        // },
        _ => (),
    })


}


unsafe fn update_push_constants
<'a>
(
    mut push_constant: PushConstants,
    delta_time: f32,
)
-> Result<PushConstants, &'a str>
{
    let view = glm::rotate
    (
        &push_constant.view,
        delta_time * 0.002,
        &glm::vec3(0.1, 0.2, 0.2),

    );

    push_constant.view = view;
    // push_constant.view = Matrix4::from_axis_angle(Vector3::new(0.2, 1.0, 0.0), Deg(0.110) * delta_time) * push_constant.view;
    Ok(push_constant)
}


unsafe fn update_uniform_buffer
(
    device: &DeviceLoader,
    uniform_transform: &mut UniformBufferObject,
    ubo_mems: &mut Vec<vk::DeviceMemory>,
    ubos: &mut Vec<vk::Buffer>,
    current_image: usize,
    delta_time: f32,
)
{
    uniform_transform.model =
        Matrix4::from_axis_angle(Vector3::new(0.0, 0.0, 0.0), Deg(0.110) * delta_time)
            * uniform_transform.model;
    let uni_transform_slice = [uniform_transform.clone()];
    let buffer_size = (std::mem::size_of::<UniformBufferObject>() * uni_transform_slice.len()) as u64;
    
    let data_ptr =
        device.map_memory(
            ubo_mems[current_image],
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
            ).expect("Failed to map memory.") as *mut UniformBufferObject;
    data_ptr.copy_from_nonoverlapping(uni_transform_slice.as_ptr(), uni_transform_slice.len());
    device.unmap_memory(ubo_mems[current_image]);
    
}



unsafe fn create_buffer
(
    device: &DeviceLoader,
    // flags: vk::BufferCreateFlags,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_type_index: u32,
    // queue_family_indices: &[u32],
) 
-> (vk::Buffer, vk::DeviceMemory) {
    let buffer_create_info = vk::BufferCreateInfoBuilder::new()
        // .flags(&[])
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&[0]);
    let buffer = device.create_buffer(&buffer_create_info, None)
        .expect("Failed to create buffer.");
    let mem_reqs = device.get_buffer_memory_requirements(buffer);
    let allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(memory_type_index);
    let buffer_memory = device
        .allocate_memory(&allocate_info, None)
        .expect("Failed to allocate memory for buffer.");
    device.bind_buffer_memory(buffer, buffer_memory, 0)
        .expect("Failed to bind buffer.");
    (buffer, buffer_memory)
}



pub fn load_model
<'a>
() 
-> Result<(Vec<VertexV3>, Vec<u32>), &'a str> 
{
    let model_path: & str = "assets/terrain__002__.obj";
    let (models, materials) = tobj::load_obj(&model_path, &tobj::LoadOptions::default()).expect("Failed to load model object!");
    let model = models[0].clone();
    let materials = materials.unwrap();
    let material = materials.clone().into_iter().nth(0).unwrap();
    let mut vertices_terr = vec![];
    let mesh = model.mesh;
    let total_vertices_count = mesh.positions.len() / 3;
    for i in 0..total_vertices_count {
        let vertex = VertexV3 {
            pos: [
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
                1.0,
            ],
            color: [0.8, 0.20, 0.30, 0.40],
        };
        vertices_terr.push(vertex);
    };
    let mut indices_terr_full = mesh.indices.clone(); 
    let mut indices_terr = vec![];
    for i in 0..(indices_terr_full.len() / 2) {
        indices_terr.push(indices_terr_full[i]);
    }
    // let mut indices = terrain_frustrum_culling(&vertices_terr, indices_terr.clone()).unwrap();


    println!("\n\nBefore {}", indices_terr.len());

    indices_terr = mesh_cull_9945(indices_terr).unwrap();
    println!("After: {}\n\n", indices_terr.len());

    Ok((vertices_terr, indices_terr))
}







unsafe fn buffer_indices
<'a>
(
    device: &DeviceLoader,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    indices: &mut Vec<u32>,
)
-> Result<(vk::Buffer), &'a str>
{
    // let vb_size = ((::std::mem::size_of_val(&(3.14 as f32))) * 9 * vertices_terr.len()) as vk::DeviceSize;
    let ib_size = (::std::mem::size_of_val(&(10 as u32)) * indices.len()) as vk::DeviceSize;
    let info = vk::BufferCreateInfoBuilder::new()
        .size(ib_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let sb = device.create_buffer(&info, None).expect("Failed to create a staging buffer.");
    let mem_reqs = device.get_buffer_memory_requirements(sb);
    let info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(2);
    let sb_mem = device.allocate_memory(&info, None).unwrap();
    device.bind_buffer_memory(sb, sb_mem, 0).unwrap();
    let data_ptr = device.map_memory(
        sb_mem,
        0,
        vk::WHOLE_SIZE,
        vk::MemoryMapFlags::empty(),
    ).unwrap() as *mut u32;
    data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
    device.unmap_memory(sb_mem);
    // Todo: add destruction if this is still working
    let info = vk::BufferCreateInfoBuilder::new()
        .size(ib_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let ib = device.create_buffer(&info, None)
        .expect("Failed to create index buffer.");
    let mem_reqs = device.get_buffer_memory_requirements(ib);
    let alloc_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(1);
    let ib_mem = device.allocate_memory(&alloc_info, None).unwrap();
    device.bind_buffer_memory(ib, ib_mem, 0);
    let info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = device.allocate_command_buffers(&info).unwrap()[0];
    let info =  vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(cb, &info).expect("Failed begin_command_buffer.");
    let info =  vk::BufferCopyBuilder::new()
        .src_offset(0)
        .dst_offset(0)
        .size(ib_size);
    device.cmd_copy_buffer(cb, sb, ib, &[info]);
    let slice = &[cb];
    device.end_command_buffer(cb).expect("Failed to end command buffer.");
    let info = vk::SubmitInfoBuilder::new()
        .wait_semaphores(&[])
        .command_buffers(slice)
        .signal_semaphores(&[]);
    device.queue_submit(queue, &[info], vk::Fence::null()).expect("Failed to queue submit.");
    Ok(ib)
}


unsafe fn buffer_vertices
<'a>
(
    device: &DeviceLoader,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    vertices: &mut Vec<VertexV3>,
)
-> Result<(vk::Buffer), &'a str>
{
    let vb_size = ((::std::mem::size_of_val(&(3.14 as f32))) * 9 * vertices.len()) as vk::DeviceSize;
    let info = vk::BufferCreateInfoBuilder::new()
        .size(vb_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let sb = device.create_buffer(&info, None).expect("Buffer create fail.");
    let mem_reqs = device.get_buffer_memory_requirements(sb);
    let info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(2);
    let sb_mem = device.allocate_memory(&info, None).unwrap();
    device.bind_buffer_memory(sb, sb_mem, 0).expect("Bind memory fail.");
    let data_ptr = device.map_memory(
        sb_mem,
        0,
        vk::WHOLE_SIZE,
        vk::MemoryMapFlags::empty(),
    ).unwrap() as *mut VertexV3;
    data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
    device.unmap_memory(sb_mem);
    let info = vk::BufferCreateInfoBuilder::new()
        .size(vb_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let vb = device.create_buffer(&info, None).expect("Create buffer fail.");
    let mem_reqs = device.get_buffer_memory_requirements(vb);
    let info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(1);
    let vb_mem = device.allocate_memory(&info, None).unwrap();
    device.bind_buffer_memory(vb, vb_mem, 0).expect("Bind memory fail.");
    let info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = device.allocate_command_buffers(&info).unwrap()[0];
    let info =  vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(cb, &info).expect("Begin command buffer fail.");
    let info = vk::BufferCopyBuilder::new()
        .src_offset(0)
        .dst_offset(0)
        .size(vb_size);
    device.cmd_copy_buffer(cb, sb, vb, &[info]);
    device.end_command_buffer(cb).expect("End command buffer fail.");
    let slice = &[cb];
    let info = vk::SubmitInfoBuilder::new()
        .wait_semaphores(&[])
        .command_buffers(slice)
        .signal_semaphores(&[]);
    device.queue_submit(queue, &[info], vk::Fence::null()).expect("Queue submit fail.");
    Ok(vb)
}


fn terrain_frustrum_culling
<'a>
(
    vertices: &Vec<VertexV3>,
    mut indices: Vec<u32>,
)
-> Result<Vec<u32>, &'a str>
{
    let max = 0.5;  // max value in the terrain mesh is 1.0 so we scale easily.
    let mut exist_bad_tri = true;
    while exist_bad_tri {
        let base: i32 = find_bad_tri(&indices, &vertices, max);
        if base == -1 {
            exist_bad_tri = false;
        } else {
            let start = base as usize;
            let end = (base + 3) as usize; 
            let check: Vec<_> = indices.drain(start..end).collect();
        }
    }
    Ok(indices)
}


fn find_bad_tri
(
    indices: &Vec<u32>,
    vertices: &Vec<VertexV3>,
    max: f32,
)
-> i32
{
    let cap = (indices.len() / 3) as i32;
    let mut base: i32 = 0;
    let mut found : bool = false;

    let x_max = 0.85;
    let y_max = 0.85;
    let z_max = 0.85;
    while !found && base < cap {
        if (vertices[indices[base as usize] as usize].pos[0].abs() > x_max) || (vertices[indices[base as usize] as usize].pos[1].abs() > y_max) || (vertices[indices[base as usize] as usize].pos[2].abs() > z_max) 
        || (vertices[indices[(base + 1) as usize] as usize].pos[0].abs() > x_max) || (vertices[indices[(base + 1) as usize] as usize].pos[1].abs() > y_max) || (vertices[indices[(base + 1) as usize] as usize].pos[2].abs() > z_max)
        || (vertices[indices[(base + 2) as usize] as usize].pos[0].abs() > x_max) || (vertices[indices[(base + 2) as usize] as usize].pos[1].abs() > y_max) || (vertices[indices[(base + 2) as usize] as usize].pos[2].abs() > z_max)
        {
            found = true;
            return base
        } else {
            base += 3;
        }
    }
    -1
}


unsafe fn record_cb_111
<'a>
(
    command_buffer: vk::CommandBuffer,
    device: &erupt::DeviceLoader,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    swapchain_image_extent: vk::Extent2D,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    indices_terr: &Vec<u32>,
    d_sets: erupt::SmallVec<vk::DescriptorSet>,
    vb: vk::Buffer,
    ib: vk::Buffer,
    push_constant: PushConstants,
)
-> Result<(), &'a str>
{
    let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();
    device.begin_command_buffer(command_buffer, &cmd_buf_begin_info).unwrap();
    let clear_values = vec![
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];
    let render_pass_begin_info = vk::RenderPassBeginInfoBuilder::new()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_image_extent,
        })
        .clear_values(&clear_values);
    
    device.cmd_begin_render_pass(
        command_buffer,
        &render_pass_begin_info,
        vk::SubpassContents::INLINE,
    );
    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
    device.cmd_bind_index_buffer(command_buffer, ib, 0, vk::IndexType::UINT32);
    device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb], &[0]);
    device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, 0, &d_sets, &[]);



    let ptr = std::ptr::addr_of!(push_constant.view) as *const c_void;

    device.cmd_push_constants
    (
        command_buffer,
        pipeline_layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        std::mem::size_of::<PushConstants>() as u32,
        ptr,
    );



    device.cmd_draw_indexed(command_buffer, (indices_terr.len()) as u32, ((indices_terr.len()) / 3) as u32, 0, 0, 0);
    device.cmd_end_render_pass(command_buffer);
    device.end_command_buffer(command_buffer).unwrap();


    Ok(())
}




unsafe fn render_pipeline_orig
<'a>
(

)
-> Result <(), &'a str>
{

    Ok(())

}


unsafe fn render_pipeline_202
<'a>
(

)
-> Result<(), &'a str>
{

    Ok(())
}



// Tesselation shader
// Secondary command buffer recorded on separate thread.
// Legacy rasterization.
unsafe fn render_pipeline_688
<'a>
(

)
-> Result<(), &'a str>
{
    Ok(())
}




// Cubes,
// Scissors
// Multiscene renders.









unsafe fn create_precursors
<'a>
(
    instance: &InstanceLoader,
    surface: vk::SurfaceKHR,
)
-> Result<(vk::PhysicalDevice, u32, vk::SurfaceFormatKHR, vk::PresentModeKHR, vk::PhysicalDeviceProperties), &'a str>
{
    let device_extensions = vec![  // trying to type the contents of this vector to be able to pass between functions..todo.
        vk::KHR_SWAPCHAIN_EXTENSION_NAME,
        vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk::KHR_RAY_QUERY_EXTENSION_NAME,
        vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk::KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        vk::KHR_SPIRV_1_4_EXTENSION_NAME,
        vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    ];

    let (physical_device, queue_family, format, present_mode, device_properties) =

    instance.enumerate_physical_devices(None)
    .unwrap()
    .into_iter()
    .filter_map(|physical_device| {
        let queue_family = match instance
            .get_physical_device_queue_family_properties(physical_device, None)
            .into_iter()
            .enumerate()
            .position(|(i, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    // (
                    //     .contains(vk::QueueFlags::GRAPHICS)
                    //     && .contains(vk::QueueFlags::TRANSFER)
                    // )
                    // .contains(vk::QueueFlags::TRANSFER)
                    && instance
                        .get_physical_device_surface_support_khr(
                            physical_device,
                            i as u32,
                            surface,
                        )
                        .unwrap()
            }) {
            Some(queue_family) => queue_family as u32,
            None => return None,
        };
        let formats = instance
            .get_physical_device_surface_formats_khr(physical_device, surface, None)
            .unwrap();
        let format = match formats
            .iter()
            .find(|surface_format| {
                (surface_format.format == vk::Format::B8G8R8A8_SRGB
                    || surface_format.format == vk::Format::R8G8B8A8_SRGB)
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
            })
            .or_else(|| formats.get(0))
        {
            Some(surface_format) => *surface_format,
            None => return None,
        };
        let present_mode = instance
            .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
            .unwrap()
            .into_iter()
            .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
            .unwrap_or(vk::PresentModeKHR::FIFO_KHR);
        let supported_device_extensions = instance
            .enumerate_device_extension_properties(physical_device, None, None)
            .unwrap();
        let device_extensions_supported =
            device_extensions.iter().all(|device_extension| {
                let device_extension = CStr::from_ptr(*device_extension);

                supported_device_extensions.iter().any(|properties| {
                    CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                })
            });
        if !device_extensions_supported {
            return None;
        }
        let device_properties = instance.get_physical_device_properties(physical_device);
        Some((
            physical_device,
            queue_family,
            format,
            present_mode,
            device_properties,
        ))
    })
    .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 0,
    })
    .expect("No suitable physical device found");

    Ok((physical_device, queue_family, format, present_mode, device_properties))

}


// There are 16k vertices in that mesh, which is killing our card with naive pipeline, 
// no multithreading, etc.  Simple function takes the indices list and just chops two out of every
// three triangles in the sequence, meaning it removes three entries from the list to remove a 
// triangle. We can also take out large regions en-masse.  
fn mesh_cull_9945
<'a>
(
    mut indices: Vec<u32>,
)
-> Result <Vec<u32>, &'a str>
{
    // let mut jdx : usize = 0;
    // let mut cool : bool = true;
    // while cool {
    //     let start = jdx as usize;
    //     let end = (jdx + 3) as usize; 
    //     indices.drain(start..end);
    //     jdx += 12;
    //     if jdx > (indices.len()) {
    //         cool = false;
    //     }
    // }


    indices.drain(12000..);
    Ok(indices)
}


// Control the view matrix with keyboard like flight controls.  Arrows for roll and pitch, z and c for yaw, pg-up and pg-down for skew/translation

fn transform_camera
<'a>
(
    camera: &mut Camera,
    view: &mut glm::Mat4,
    control_input: &mut ControlInput,
)
-> Result <(), &'a str>
{

    let scalar_45 = 0.03;

    // camera up is the yaw axis.
    // So that's what we keep transforming.
    let orientation: glm::Vec3 = glm::normalize(&(camera.target - camera.location));
    // Orientation is the roll axis.  with the direction of positive skew.
    // let pitch_axis = glm::cross(orientation, &camera.up);

    let view_rolled = glm::rotate(&view, (control_input.roll as f32) * scalar_45, &orientation);
    // this transforms the whole view matrix with rotation over the roll axis, called orientation.


    camera.up = glm::rotate_vec3(&camera.up, (control_input.roll as f32) * scalar_45, &orientation);  // this transforms the yaw axis by the rotation matrix.


    // Pitch axis is up-vector/yax-axis crossed with orientation-vector/roll-axis.
    let pitch_axis: glm::Vec3 = glm::normalize(&glm::cross(&orientation, &camera.up));  
    // We may as well just store this memoize it, and transform as needed.

    // The axis for the pitch can be gotten from the cross product of the orientation vector with...

    let view_pitched = glm::rotate(&view_rolled, (control_input.pitch as f32) * scalar_45, &pitch_axis);

    // let view = view_yawwed...
    *view = glm::rotate(&view_pitched, (control_input.yaw as f32) * scalar_45, &camera.up);  // May need to do this before modifying that vector?



    // let = camera.location - camera.orientation


    // Okay actually what we should be doing is maintaining independent state of the view object,
    // in a struct,  This feeds the look_at function.

    // println!("view: {:?}", view);

    // First we need to inspect the view matrix given to determine the vector normal of the camera view.  This is 
    // Since our model is at the origin, the view vector of the camera since it's pointing at the origin is the 
    // situational point of the camera, but negative of that.  so if the camera went up to .5, .5, .5, it's pointed at the origin,
    // but the vector of it's view axis is negative where it is.  

    // if the model was at 1,1,1


    // camera at 1, -1.  target at 1, 1.  
    // so the orientation vector not normalized should be 0, 2, which comes from camera - target.  

    
    
    *control_input = ControlInput {
        roll: 0,
        pitch: 0,
        yaw: 0,
        skew: 0
    };


    // roll would be rotation around the axis normal of the camera look at point.
    // pitch would be a rotation around one axis perpendicular to the camera view.
    // yaw is rotation around the other axis perpindicular to the camera view.
    // translation will be through the camera's view.
    

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;





    #[test]
    fn test_transform_camera() {
        let mut control_input = ControlInput {
            roll: 0,
            pitch: 0,
            yaw: 0,
            skew: 0,
        };
        let scalar_33 = 100000000.0;
        let mut camera = Camera {
            location: glm::vec3(1.0 / scalar_33, 1.0 / scalar_33, 1.0 / scalar_33),
            target: glm::vec3(0.0, 0.0, 0.0),
            up: glm::vec3(0.0, 1.0, 0.0),
        };
        let mut view: glm::Mat4 = glm::look_at::<f32>
        (
            &camera.location,
            &camera.target,
            &camera.up,
        );
        let result = transform_camera(&mut camera, &mut view, &mut control_input).unwrap();
        println!("result: {:?}", result);
        assert_eq!(3, 3);
    }

    #[test]
    fn test_new_camera() {
        let scalar_33 = 100000000.0;
        let camera_location = glm::vec3(1.0 / scalar_33, 1.0 / scalar_33, 1.0 / scalar_33);
        let image_target = glm::vec3(0.0, 0.0, 0.0);

        let roll_axis_normal: glm::Vec3 = glm::normalize(&(camera_location - image_target));
        let yaw_axis_normal: glm::Vec3 = glm::vec3(0.0, 1.0, 0.0);
        let pitch_axis_normal: glm::Vec3 = glm::cross(&roll_axis_normal, &yaw_axis_normal);



        let mut camera = NewCamera {
            position: camera_location,
            attitude: Attitude {
                roll_axis_normal,
                pitch_axis_normal,
                yaw_axis_normal,
            }
        };




    }

}



#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Attitude {
    // logically, the third axis normal can be derived from the other two, memoization indicates the third.
    roll_axis_normal: glm::Vec3,
    pitch_axis_normal: glm::Vec3,
    yaw_axis_normal: glm::Vec3,
}




#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NewCamera {
    attitude: Attitude,
    position: glm::Vec3,
}