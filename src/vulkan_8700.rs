#![feature(drain_filter)]
// use super::precursors::*;
// use super::pipeline_101::*;
// use super::pipeline_102::*;
use super::utilities::*;
use super::buffer_ops::buffer_indices::*;
use super::buffer_ops::buffer_vertices::*;
use super::buffer_ops::create_buffer::*;
use super::buffer_ops::update_uniform_buffer::*;
use super::pipelines::pipeline_101::*;
use super::command_buffers::record_cb_218::*;
use super::command_buffers::record_cb_219::*;
// use super::spatial_transforms::camera::*;

use crate::spatial_transforms::camera::*;
use crate::data_structures::vertex_v3::VertexV3;

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
    // sync::mpsc::channel,
    os::raw::c_char,
    ptr,
    result::Result,
    result::Result::*,
    string::String,
    sync::{Arc, Mutex, mpsc, mpsc::{channel}},
    thread,
    time,
};
// use std::sync::mpsc;
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

use closure::closure;
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

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    validation_layers: bool,
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




pub unsafe fn vulkan_routine_8700
()
{
    let opt = Opt { validation_layers: false };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .with_resizable(false)
        .with_maximized(true)
        .build(&event_loop)
        .unwrap();
    let entry = Arc::new(EntryLoader::new().unwrap());
    let application_name = CString::new("Vulkan-Routine-8700").unwrap();
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
    // let device__pre  = Arc::new(Mutex::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap()));
    // let device = Arc::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap());
    let device  = Arc::new(Mutex::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap()));

    // device.lock().unwrap();
    // let device = device__pre.lock().unwrap();
    
    let queue = device.lock().unwrap().get_device_queue(queue_family, 0);

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
    let swapchain = device.lock().unwrap().create_swapchain_khr(&swapchain_info, None).unwrap();
    let swapchain_images = device.lock().unwrap().get_swapchain_images_khr(swapchain, None).unwrap();
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
            device.lock().unwrap().create_image_view(&image_view_info, None).unwrap()
        })
        .collect();

    let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = device.lock().unwrap().create_command_pool(&command_pool_info, None).unwrap();
    // original, still using



    let command_pools : Arc<Mutex<Vec<Arc<Mutex<vk::CommandPool>>>>> = Arc::new(Mutex::new(vec![]));

    
    for _ in 0..swapchain_images.len() {
        let info = vk::CommandPoolCreateInfoBuilder::new()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family);
        let command_pool = Arc::new(Mutex::new(device.lock().unwrap().create_command_pool(&info, None).unwrap()));
        command_pools.lock().unwrap().push(command_pool);
        // command_pools.push(command_pool);
    }
    
    let (mut vertices_terr, mut indices_terr) = load_model().unwrap();

    let grid_factor = 10;
    let grid_resolution = 1.0 / grid_factor as f32;
    let mut vertices_grid = vec![];

    for idx in 0..grid_factor {
        for jdx in 0..grid_factor {
            for kdx in 0..grid_factor {
                let vertex = VertexV3 {
                    pos: [
                        (grid_resolution * (idx as f32)),
                        (grid_resolution * (jdx as f32)),
                        (grid_resolution * (kdx as f32)),
                        1.0,
                    ],
                    color: [0.913, 0.9320, 0.80, 0.90],
                };
                vertices_grid.push(vertex);
            }
        }
    }

    let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);
    let ib = buffer_indices
    (
        device.clone(),
        queue,
        command_pool,
        &mut indices_terr,
    ).unwrap();

    let vb = buffer_vertices
    (
        device.clone(),
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
    let bindings = &[binding];
    let info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty()) 
        .bindings(bindings);
    let descriptor_set_layout = device.lock().unwrap().create_descriptor_set_layout(&info, None).unwrap();

    let ubo_size = ::std::mem::size_of::<UniformBufferObject>();
    let mut uniform_buffers: Vec<vk::Buffer> = vec![];
    let mut uniform_buffers_memories: Vec<vk::DeviceMemory> = vec![];
    let swapchain_image_count = swapchain_images.len();

    for _ in 0..swapchain_image_count {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            device.clone(),
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

    let scalar_33 = 100000.0;
    let camera_location = glm::vec3(1.0 / scalar_33, 1.0 / scalar_33, 1.0 / scalar_33);
    let image_target = glm::vec3(0.0, 0.0, 0.0);

    let roll_axis_normal: glm::Vec3 = glm::normalize(&(camera_location - image_target));
    let yaw_axis_normal: glm::Vec3 = glm::vec3(0.0, 1.0, 0.0);  // Also known as the 'up' vector.
    let pitch_axis_normal: glm::Vec3 = glm::cross(&roll_axis_normal, &yaw_axis_normal);

    let mut camera = Camera {
        position: camera_location,
        attitude: Attitude {
            roll_axis_normal,
            pitch_axis_normal,
            yaw_axis_normal, 
        }
    };

    let camera_2 = Arc::new(Mutex::new(Camera {
        position: camera_location,
        attitude: Attitude {
            roll_axis_normal,
            pitch_axis_normal,
            yaw_axis_normal, 
        }
    }));


    let mut view: glm::Mat4 = glm::look_at::<f32>
    (
        &camera_location,
        &image_target,
        &yaw_axis_normal,
    );


    let pc_view: Arc<Mutex<glm::Mat4>> = Arc::new(Mutex::new(glm::look_at::<f32>
    (
        &camera_location,
        &image_target,
        &yaw_axis_normal,
    )));



    let pool_size = vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(swapchain_image_count as u32);
    let pool_sizes = &[pool_size];
    let set_layouts = &[descriptor_set_layout];
    let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(pool_sizes)
        .max_sets(swapchain_image_count as u32);

    let desc_pool = device.lock().unwrap().create_descriptor_pool(&pool_info, None).unwrap();
    let d_set_alloc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(set_layouts);

    let d_sets = device.lock().unwrap().allocate_descriptor_sets(&d_set_alloc_info).expect("failed in alloc DescriptorSet");
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

        device.lock().unwrap().update_descriptor_sets(&d_write_sets, &[]);
        update_uniform_buffer
        (
            device.clone(), 
            &mut uniform_transform, 
            &mut uniform_buffers_memories, 
            &mut uniform_buffers, 
            i as usize, 
            2.3
        );
    }

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

    let rp_pre = device.lock().unwrap().create_render_pass(&render_pass_info, None).unwrap();
    let render_pass = Arc::new(Mutex::new(
        device.lock().unwrap().create_render_pass(&render_pass_info, None).unwrap()
    ));

    let (
        pipeline,
        pipeline_layout,
        depth_image_view,
        shader_vert,
        shader_frag,
    ) = pipeline_101
    (
        device.clone(),
        render_pass.clone(),
        &format,
        &swapchain_image_extent,
    ).unwrap();

    let swapchain_framebuffers: Vec<_> = swapchain_image_views
        .iter()
        .map(|image_view| {
            let attachments = vec![*image_view, depth_image_view];
            let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
                .render_pass(*render_pass.lock().unwrap())
                .attachments(&attachments)
                .width(swapchain_image_extent.width)
                .height(swapchain_image_extent.height)
                .layers(1);
            device.lock().unwrap().create_framebuffer(&framebuffer_info, None).unwrap()
        })
        .collect();

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let cmd_bufs = device.lock().unwrap().allocate_command_buffers(&cmd_buf_allocate_info).unwrap();

    let cb_2_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::SECONDARY)
        .command_buffer_count(swapchain_framebuffers.len() as _);
    let cb_2s = device.lock().unwrap().allocate_command_buffers(&cb_2_info).unwrap();
    let mut primary_command_buffers: Vec<vk::CommandBuffer> = vec![];
    let mut secondary_command_buffers: Vec<vk::CommandBuffer> = vec![];
    for img_idx in 0..swapchain_framebuffers.len() {
        let command_pool_arc = &command_pools.lock().unwrap()[img_idx];
        let command_pool = command_pool_arc.lock().unwrap();

        
        let primary_cb_alloc_info = vk::CommandBufferAllocateInfoBuilder::new()

            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        primary_command_buffers.push(device.lock().unwrap().allocate_command_buffers(&primary_cb_alloc_info).unwrap()[0]);
    //     let secondary_cb_alloc_info = vk::CommandBufferAllocateInfoBuilder::new()
    //         .command_pool(command_pool)
    //         .level(vk::CommandBufferLevel::SECONDARY)
    //         .command_buffer_count(1);
    //     secondary_command_buffers.push(device.lock().unwrap().allocate_command_buffers(&secondary_cb_alloc_info).unwrap()[0]);
    }

    let now = Instant::now();
    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let image_available_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| device.lock().unwrap().create_semaphore(&semaphore_info, None).unwrap())
        .collect();
    let render_finished_semaphores: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| device.lock().unwrap().create_semaphore(&semaphore_info, None).unwrap())
        .collect();
    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fences: Vec<_> = (0..FRAMES_IN_FLIGHT)
        .map(|_| device.lock().unwrap().create_fence(&fence_info, None).unwrap())
        .collect();
    let mut images_in_flight: Vec<_> = swapchain_images.iter().map(|_| vk::Fence::null()).collect();
    let mut frame = 0;



    let (rcb_tx, rcb_rx) : (mpsc::Sender<u8>, mpsc::Receiver<u8>) = channel();

    let rcb_closure = closure!(
        clone rcb_tx,
        clone device,
        clone render_pass,
        

        ||
        {


            // The main thread will also use this transmitter.  This rcb record command buffer thread will signal when it is done recording command buffer.
            // Actually it doesn't even need to do that.  


        }
    );


    thread::spawn(rcb_closure);



    let (tx, rx) : (mpsc::Sender<u8>, mpsc::Receiver<u8>) = channel();


    let state_thread_closure = closure!(
        move rx,
        clone pc_view,
        move camera_2,
        ||
        {


            
            println!("Hello state management thread.");
            let scalar_45 = 0.345;
            let mut counter = 0;
            while true {
                let mut attitude = camera_2.lock().unwrap().attitude;
                println!("check attitude: {:?}", attitude);
                let mut position = camera_2.lock().unwrap().position;
                // attitude.roll_axis_normal = glm::rotate_vec3(&attitude.roll_axis_normal, scalar_45, &attitude.roll_axis_normal);                
                // .lock().unwrap().attitude = attitude;
                // match rx.recv().unwrap() {
                //     0 => *attitude.roll_axis_normal = *glm::rotate_vec3(&attitude.roll_axis_normal, scalar_45, &attitude.roll_axis_normal),
                //     1 => *attitude.roll_axis_normal = *glm::rotate_vec3(&attitude.roll_axis_normal, -scalar_45, &attitude.roll_axis_normal),
                //     2 => *attitude.pitch_axis_normal = *glm::rotate_vec3(&attitude.pitch_axis_normal, scalar_45, &attitude.pitch_axis_normal),
                //     3 => *attitude.pitch_axis_normal = *glm::rotate_vec3(&attitude.pitch_axis_normal, -scalar_45, &attitude.pitch_axis_normal),
                //     4 => *attitude.yaw_axis_normal = *glm::rotate_vec3(&attitude.yaw_axis_normal, scalar_45, &attitude.yaw_axis_normal),
                //     5 => *attitude.yaw_axis_normal = *glm::rotate_vec3(&attitude.yaw_axis_normal, scalar_45, &attitude.yaw_axis_normal),
                //     _ => println!(" nothing "),
                // }

                let ans = rx.recv().unwrap();

               


                
                let mut data_ref = pc_view.lock().unwrap();

                println!("data_ref {:?}", data_ref);

                let cursor = glm::look_at::<f32>
                (
                    &position,
                    &(&position + &attitude.roll_axis_normal),
                    &attitude.yaw_axis_normal,
                );

                *data_ref = cursor;
                
                let modded = glm::rotate_y(&cursor, 0.053 * (counter as f32));
                *data_ref = modded; 

                counter += 1;

            }
        }
    );

    thread::spawn(state_thread_closure);

    



    // let rec_cb_closure = closure!(
    //     clone device,
    //     clone record_cb_219,
    //     ||
    // {
    //     // println!("{:?} {:?}", record_cb_218, device);
    // });


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
                },
                (winit::event::VirtualKeyCode::Right, ElementState::Pressed) => {
                    tx.send(0).unwrap();
                },
                (winit::event::VirtualKeyCode::Left, ElementState::Pressed) => {
                    tx.send(1).unwrap();
                },
                (winit::event::VirtualKeyCode::Up, ElementState::Pressed) => {
                    tx.send(2).unwrap();
                },
                (winit::event::VirtualKeyCode::Down, ElementState::Pressed) => {
                    tx.send(3).unwrap();
                },
                (winit::event::VirtualKeyCode::Semicolon, ElementState::Pressed) => {
                    tx.send(4).unwrap();
                },
                (winit::event::VirtualKeyCode::Q, ElementState::Pressed) => {
                    tx.send(5).unwrap();
                },
                _ => (),

            },
            _ => (),
        },
        Event::MainEventsCleared => {
            device.lock().unwrap().wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX).unwrap();
            let image_index = device.lock().unwrap().acquire_next_image_khr
            (
                swapchain,
                u64::MAX,
                image_available_semaphores[frame],
                vk::Fence::null(),
            ).unwrap();
            let delta_time = now.elapsed().as_secs_f32();

            let image_in_flight = images_in_flight[image_index as usize];
            if !image_in_flight.is_null() {
                device.lock().unwrap().wait_for_fences(&[image_in_flight], true, u64::MAX).unwrap();
            }
            images_in_flight[image_index as usize] = in_flight_fences[frame];
            let wait_semaphores = vec![image_available_semaphores[frame]];
            
            let command_pool_arc = &command_pools.lock().unwrap()[image_index as usize];




            let command_buffer = cmd_bufs[image_index as usize];
            let cb_2 = cb_2s[image_index as usize];
            let framebuffer = swapchain_framebuffers[image_index as usize];


            // In the multi-threaded version, we should have be waiting on the previous cb_recording.

            // rx_34.revd

            let cb_34 = record_cb_219
            (
                device.clone(),
                render_pass.clone(),
                command_pool_arc,
                pipeline, // primary_pipeline,
                pipeline_layout,
                &mut primary_command_buffers,
                &mut secondary_command_buffers,
                &framebuffer,
                image_index,
                swapchain_image_extent,
                &indices_terr,
                d_sets.clone(),
                vb,
                ib,
                *pc_view.lock().unwrap(),
            ).unwrap();

            let cbs_35 = [cb_34];
            let signal_semaphores = vec![render_finished_semaphores[frame]];
            let submit_info = vk::SubmitInfoBuilder::new()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(&cbs_35)
                .signal_semaphores(&signal_semaphores);

            let in_flight_fence = in_flight_fences[frame];
            device.lock().unwrap().reset_fences(&[in_flight_fence]).unwrap();
            device.lock().unwrap()
                .queue_submit(queue, &[submit_info], in_flight_fence)
                .unwrap();
            let swapchains = vec![swapchain];
            let image_indices = vec![image_index];
            let present_info = vk::PresentInfoKHRBuilder::new()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            device.lock().unwrap().queue_present_khr(queue, &present_info).unwrap();
            frame = (frame + 1) % FRAMES_IN_FLIGHT;
        }
        Event::LoopDestroyed => unsafe {
            device.lock().unwrap().device_wait_idle().unwrap();
            for &semaphore in image_available_semaphores
                .iter()
                .chain(render_finished_semaphores.iter())
            {
                device.lock().unwrap().destroy_semaphore(semaphore, None);
            }
            for &fence in &in_flight_fences {
                device.lock().unwrap().destroy_fence(fence, None);
            }
            device.lock().unwrap().destroy_command_pool(command_pool, None);
            for &framebuffer in &swapchain_framebuffers {
                device.lock().unwrap().destroy_framebuffer(framebuffer, None);
            }
            device.lock().unwrap().destroy_pipeline(pipeline, None);
            device.lock().unwrap().destroy_render_pass(*render_pass.lock().unwrap(), None);
            device.lock().unwrap().destroy_pipeline_layout(pipeline_layout, None);
            device.lock().unwrap().destroy_shader_module(shader_vert, None);
            device.lock().unwrap().destroy_shader_module(shader_frag, None);
            for &image_view in &swapchain_image_views {
                device.lock().unwrap().destroy_image_view(image_view, None);
            }
            device.lock().unwrap().destroy_swapchain_khr(swapchain, None);
            device.lock().unwrap().destroy_device(None);
            instance.destroy_surface_khr(surface, None);
            // instance.destroy_debug_utils_messenger_ext(messenger, None);
            // if !messenger.is_null() {
            //     instance.destroy_debug_utils_messenger_ext(messenger, None);
            // }
            instance.destroy_instance(None);
            println!("Exited cleanly");
        },
        _ => (),
    })









}




fn mesh_cull_9945
<'a>
(
    mut indices: Vec<u32>,
)
-> Result <Vec<u32>, &'a str>
{
    indices.drain(20000..);
    Ok(indices)
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
    println!("\n\nBefore {}", indices_terr.len());
    indices_terr = mesh_cull_9945(indices_terr).unwrap();
    println!("After: {}\n\n", indices_terr.len());
    Ok((vertices_terr, indices_terr))
}
