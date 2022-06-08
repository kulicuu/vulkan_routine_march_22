// A routine for testing multi-threaded rendering.


#![feature(drain_filter)]

use super::precursors::*;
use super::pipeline_101::*;
use super::pipeline_102::*;

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
    sync::{Arc, Mutex, mpsc, mpsc::channel, Condvar},
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
const TITLE: &str = "vulkan-routine-6400";
const FRAMES_IN_FLIGHT: usize = 2;
const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");
const SHADER_VERT: &[u8] = include_bytes!("../spv/s_400_.vert.spv");
const SHADER_FRAG: &[u8] = include_bytes!("../spv/s1.frag.spv");


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
struct ControlInput {
    roll: i32,
    pitch: i32,
    yaw: i32,
    skew: i32,
}


// There is a plan that envisions three threads. One for state management, 
// consuming messages carrying async input updates, for example keypresses.
// Typically this first thread is doing maths, and as the program gets more complex,
// these responsibilities would spawn more thread architecture.
// Two is just for recording command buffers.
// Zero is the main, thread, organizing the program.





pub unsafe fn vulkan_routine_6400
()
{
    println!("\n6400\n");


    println!("\n6300\n");
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

    let (
        physical_device, 
        queue_family, 
        format, 
        present_mode, 
        device_properties
    ) = create_precursors
    (
        &instance,
        surface.clone(),
    ).unwrap();


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
    // original, still using
    let mut command_pools: Vec<vk::CommandPool> = vec![]; // this follows usage in the tutorial  https://github.com/KyleMayes/vulkanalia/blob/master/tutorial/src/32_secondary_command_buffers.rs

    for _ in 0..swapchain_images.len() {
        let info = vk::CommandPoolCreateInfoBuilder::new()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family);
        let command_pool = device.create_command_pool(&info, None).unwrap();
        command_pools.push(command_pool);
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


    // let ib = buffer_indices
    // (
    //     &device,
    //     queue,
    //     command_pool,
    //     &mut indices_terr,
    // ).unwrap();
    
    // let vb = buffer_vertices
    // (
    //     &device,
    //     queue,
    //     command_pool,
    //     &mut vertices_terr, 
    // ).unwrap();

    // let vb_grid = buffer_vertices
    // (
    //     &device,
    //     queue,
    //     command_pool,
    //     &mut vertices_grid,
    // ).unwrap();

    // let info = vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
    //     .binding_flags(&[vk::DescriptorBindingFlags::empty()]);

    // let samplers = [vk::Sampler::null()];
    // let binding = vk::DescriptorSetLayoutBindingBuilder::new()
    //     .binding(0)
    //     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    //     .descriptor_count(1)
    //     .stage_flags(vk::ShaderStageFlags::VERTEX)
    //     .immutable_samplers(&samplers);
    // let bindings = &[binding];
    // let info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
    //     .flags(vk::DescriptorSetLayoutCreateFlags::empty()) 
    //     .bindings(bindings);
    // let descriptor_set_layout = device.create_descriptor_set_layout(&info, None).unwrap();


//999999--------------------------------------------------------






    let (tx, rx) = channel();

    let (tx2, rx2) = channel();

    let pair = Arc::new((Mutex::new(false), Condvar::new()));

    thread::spawn(closure!(clone pair, || {
        // let &(ref lock, ref cvar) = &*pair;
        // let mut started = lock.lock().unwrap();
        // *started = true;
        // // We notify the condvar that the value has changed.
        // cvar.notify_one();


        tx.send(10).unwrap();



        println!("device {:?}", device);

        unsafe fn record_cb_111
        <'a>
        (
            command_pool: vk::CommandPool,
            command_buffer: vk::CommandBuffer,
            cb_2: vk::CommandBuffer,
            device: &erupt::DeviceLoader,
            render_pass: vk::RenderPass,
            framebuffer: vk::Framebuffer,
            swapchain_image_extent: vk::Extent2D,
            pipeline: vk::Pipeline,
            pipeline_layout: vk::PipelineLayout,
            pipeline_grid: vk::Pipeline,
            pipeline_layout_grid: vk::PipelineLayout,
            indices_terr: &Vec<u32>,
            d_sets: erupt::SmallVec<vk::DescriptorSet>,
            vb: vk::Buffer,
            vb_grid: vk::Buffer,
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
    }));


    // thread::spawn(move || {
    //     tx.send(10).unwrap();



    //     println!("device {:?}", device);

    //     unsafe fn record_cb_111
    //     <'a>
    //     (
    //         command_pool: vk::CommandPool,
    //         command_buffer: vk::CommandBuffer,
    //         cb_2: vk::CommandBuffer,
    //         device: &erupt::DeviceLoader,
    //         render_pass: vk::RenderPass,
    //         framebuffer: vk::Framebuffer,
    //         swapchain_image_extent: vk::Extent2D,
    //         pipeline: vk::Pipeline,
    //         pipeline_layout: vk::PipelineLayout,
    //         pipeline_grid: vk::Pipeline,
    //         pipeline_layout_grid: vk::PipelineLayout,
    //         indices_terr: &Vec<u32>,
    //         d_sets: erupt::SmallVec<vk::DescriptorSet>,
    //         vb: vk::Buffer,
    //         vb_grid: vk::Buffer,
    //         ib: vk::Buffer,
    //         push_constant: PushConstants,
    //     )
    //     -> Result<(), &'a str>
    //     {
    //         let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();
    //         device.begin_command_buffer(command_buffer, &cmd_buf_begin_info).unwrap();
    //         let clear_values = vec![
    //             vk::ClearValue {
    //                 color: vk::ClearColorValue {
    //                     float32: [0.0, 0.0, 0.0, 1.0],
    //                 },
    //             },
    //             vk::ClearValue {
    //                 depth_stencil: vk::ClearDepthStencilValue {
    //                     depth: 1.0,
    //                     stencil: 0,
    //                 },
    //             },
    //         ];
    //         let render_pass_begin_info = vk::RenderPassBeginInfoBuilder::new()
    //             .render_pass(render_pass)
    //             .framebuffer(framebuffer)
    //             .render_area(vk::Rect2D {
    //                 offset: vk::Offset2D { x: 0, y: 0 },
    //                 extent: swapchain_image_extent,
    //             })
    //             .clear_values(&clear_values);
    //         device.cmd_begin_render_pass(
    //             command_buffer,
    //             &render_pass_begin_info,
    //             vk::SubpassContents::INLINE,
    //         );
    //         device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
    //         device.cmd_bind_index_buffer(command_buffer, ib, 0, vk::IndexType::UINT32);
    //         device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb], &[0]);
    //         device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, 0, &d_sets, &[]);
    //         let ptr = std::ptr::addr_of!(push_constant.view) as *const c_void;
    //         device.cmd_push_constants
    //         (
    //             command_buffer,
    //             pipeline_layout,
    //             vk::ShaderStageFlags::VERTEX,
    //             0,
    //             std::mem::size_of::<PushConstants>() as u32,
    //             ptr,
    //         );
    //         device.cmd_draw_indexed(command_buffer, (indices_terr.len()) as u32, ((indices_terr.len()) / 3) as u32, 0, 0, 0);
    //         device.cmd_end_render_pass(command_buffer);
    //         device.end_command_buffer(command_buffer).unwrap();
    //         Ok(())
    //     }
    // });

    assert_eq!(rx.recv().unwrap(), 10);


    tx2.send(15);
    tx2.send(167);







    // let ib = buffer_indices
    // (
    //     &device,
    //     queue,
    //     command_pool,
    //     &mut indices_terr,
    // ).unwrap();
    
    // let vb = buffer_vertices
    // (
    //     &device,
    //     queue,
    //     command_pool,
    //     &mut vertices_terr, 
    // ).unwrap();

    // let vb_grid = buffer_vertices
    // (
    //     &device,
    //     queue,
    //     command_pool,
    //     &mut vertices_grid,
    // ).unwrap();

    // let info = vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
    //     .binding_flags(&[vk::DescriptorBindingFlags::empty()]);

    // let samplers = [vk::Sampler::null()];
    // let binding = vk::DescriptorSetLayoutBindingBuilder::new()
    //     .binding(0)
    //     .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    //     .descriptor_count(1)
    //     .stage_flags(vk::ShaderStageFlags::VERTEX)
    //     .immutable_samplers(&samplers);
    // let bindings = &[binding];
    // let info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
    //     .flags(vk::DescriptorSetLayoutCreateFlags::empty()) 
    //     .bindings(bindings);
    // let descriptor_set_layout = device.create_descriptor_set_layout(&info, None).unwrap();











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
    indices.drain(15000..);
    Ok(indices)
}



// three threads.
// one main thread.

// one math thread

// one command buffer recording thread.


