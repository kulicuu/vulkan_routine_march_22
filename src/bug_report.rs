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
    // let app_info = vk::ApplicationInfoBuilder::new()
    //     .application_name(&application_name)
    //     .application_version(vk::make_api_version(0, 1, 0, 0))
    //     .engine_name(&engine_name)
    //     .engine_version(vk::make_api_version(0, 1, 0, 0))
    //     .api_version(vk::make_api_version(0, 1, 0, 0));
    // let mut instance_extensions = surface::enumerate_required_extensions(&window).unwrap();



//999999--------------------------------------------------------






    let (tx, rx) = channel();

    let (tx2, rx2) = channel();

    let pair = Arc::new((Mutex::new(false), Condvar::new()));



    // struct RecordCBState<'a> {

    //     command_pool: vk::CommandPool,  // this could 
    //     command_buffer: vk::CommandBuffer,
    //     cb_2: vk::CommandBuffer,
    //     device: &erupt::DeviceLoader,
    //     render_pass: vk::RenderPass,
    //     framebuffer: vk::Framebuffer,
    //     swapchain_image_extent: vk::Extent2D,
    //     pipeline: vk::Pipeline,
    //     pipeline_layout: vk::PipelineLayout,
    //     pipeline_grid: vk::Pipeline,
    //     pipeline_layout_grid: vk::PipelineLayout,
    //     indices_terr: &Vec<u32>,
    //     d_sets: erupt::SmallVec<vk::DescriptorSet>,
    //     vb: vk::Buffer,
    //     vb_grid: vk::Buffer,
    //     ib: vk::Buffer,
    //     push_constant: PushConstants,

    // }
    

    thread::spawn(closure!(clone pair, || {
        // let &(ref lock, ref cvar) = &*pair;
        // let mut started = lock.lock().unwrap();
        // *started = true;
        // // We notify the condvar that the value has changed.
        // cvar.notify_one();


        // tx.send(10).unwrap();



        // println!("device {:?}", device);

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

















}


