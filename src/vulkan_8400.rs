// #![feature(drain_filter)]

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
    sync::{Arc, Barrier, Mutex, mpsc},
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
const TITLE: &str = "vulkan-routine-8400";
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


#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct OldCamera {
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


pub unsafe fn vulkan_routine_8400
()
{
    println!("\nSTART________________________________8400\n");
    let opt = Opt { validation_layers: false };
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

    let device = Arc::new(Mutex::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap()));



    let (tx, rx): (mpsc::Sender<u32>, mpsc::Receiver<u32>) = mpsc::channel();




    // It could be a good pattern to offload the whole building operation to a thread.
    // Then will wait for it to join before proceeding.


    // Then for the cb_record_thread_closure


    let vulkan_startup_thread_closure = closure!()









    let c1 = closure!(clone device, move rx, || {
        println!("\n\nSTART __________________________Closure-1-------\n\n");




        // device;    
        // let queue = device.get_device_queue(queue_family, 0);


        // https://docs.rs/closure/0.3.0/closure/
        // let &(ref lock, ref cvar) = &*pair;
        // let mut started = lock.lock().unwrap();
        // *started = true;
        // // We notify the condvar that the value has changed.
        // cvar.notify_one();

        let device = &*device;


        let queue = device.lock().unwrap().get_device_queue(2, 0);


        println!("receiver recv {:?}", rx.recv().unwrap());

        println!("\n\n_____________________Closure-1-------______END\n\n");
        true
    });


    thread::spawn(c1);

    // let thread_closure = closure!(move string, ref x, ref mut y, clone rc, |arg: i32| {

    // let thread_closure = closure!(clone device, || {
    //     false
    // };

    // thread::spawn(clone thread_closure);



    tx.send(33).unwrap();
    // thread::sleep(Duration::from_millis(250));


    println!("FLow 3838");



    unsafe fn record_cb_1131
    <'a>
    (

    )
    -> Result<(), &'a str>
    {

    }


    println!("\n\n------------Routine------------8400------------END------\n\n")
}







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



    // device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty()).unwrap();
    
    // let inheritance_info = vk::CommandBufferInheritanceInfoBuilder::new()
    //     .render_pass(render_pass)
    //     .subpass(0)
    //     .framebuffer(framebuffer);

    // let cb_2_begin_info = vk::CommandBufferBeginInfoBuilder::new()
    //     .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
    //     .inheritance_info(&inheritance_info);
    // device.begin_command_buffer(cb_2, &cb_2_begin_info).unwrap();



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
    // device.cmd_execute_commands(command_buffer, &[cb_2]);
    device.cmd_end_render_pass(command_buffer);
    device.end_command_buffer(command_buffer).unwrap();
    Ok(())
}