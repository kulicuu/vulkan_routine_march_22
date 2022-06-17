use super::record_cb_218::*;

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
    sync::{Arc, Mutex, mpsc},
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


// #[repr(C)]
// #[derive(Clone, Debug, Copy)]
// pub struct PushConstants {
//     pub view: glm::Mat4,
// }


pub unsafe fn record_cb_219
<'a>
(
    device: Arc<Mutex<DeviceLoader>>,
    render_pass: Arc<Mutex<vk::RenderPass>>,
    command_pool: &Arc<Mutex<vk::CommandPool>>,
    primary_pipeline: vk::Pipeline,
    primary_pipeline_layout: vk::PipelineLayout,
    primary_command_buffers: &mut Vec<vk::CommandBuffer>,
    secondary_command_buffers: &mut Vec<vk::CommandBuffer>,
    framebuffer: &vk::Framebuffer,
    image_index: u32,
    swapchain_image_extent: vk::Extent2D,
    indices_terr: &Vec<u32>,
    d_sets: erupt::SmallVec<vk::DescriptorSet>,
    vb: vk::Buffer,
    ib: vk::Buffer,
    pc_view: glm::Mat4,

)
-> Result<(vk::CommandBuffer), &'a str>
{
    
    

    device.lock().unwrap().reset_command_pool(*command_pool.lock().unwrap(), vk::CommandPoolResetFlags::empty()).unwrap();

    // Primary command buffer:
    let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(*command_pool.lock().unwrap())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let primary_cb = device.lock().unwrap().allocate_command_buffers(&allocate_info).unwrap()[0];
    primary_command_buffers[image_index as usize] = primary_cb;


    let pri_cb_begin_info = vk::CommandBufferBeginInfoBuilder::new();
    device.lock().unwrap().begin_command_buffer(primary_cb, &pri_cb_begin_info).unwrap();
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
        .render_pass(*render_pass.lock().unwrap())
        .framebuffer(*framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_image_extent,
        })
        .clear_values(&clear_values);
    device.lock().unwrap().cmd_begin_render_pass(
        primary_cb,
        &render_pass_begin_info,
        // vk::SubpassContents::INLINE,
        vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
    );
    device.lock().unwrap().cmd_bind_pipeline(primary_cb, vk::PipelineBindPoint::GRAPHICS, primary_pipeline);
    device.lock().unwrap().cmd_bind_index_buffer(primary_cb, ib, 0, vk::IndexType::UINT32);
    device.lock().unwrap().cmd_bind_vertex_buffers(primary_cb, 0, &[vb], &[0]);
    device.lock().unwrap().cmd_bind_descriptor_sets(primary_cb, vk::PipelineBindPoint::GRAPHICS, primary_pipeline_layout, 0, &d_sets, &[]);
    // let ptr = std::ptr::addr_of!(push_constant.view) as *const c_void;
    let ptr = std::ptr::addr_of!(pc_view) as *const c_void;
    println!("pc_view {:?}", pc_view);
    // let ptr_2 = std::ptr::addr_of!(*pc_view.lock().unwrap()) as *const c_void;
    // let ptr_2 = std::ptr::addr_of!(pc_view) as *const c_void;
    device.lock().unwrap().cmd_push_constants
    (
        primary_cb,
        primary_pipeline_layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        std::mem::size_of::<PushConstants>() as u32,
        // ptr_2,
        ptr,
    );
    device.lock().unwrap().cmd_draw_indexed(primary_cb, (indices_terr.len()) as u32, ((indices_terr.len()) / 3) as u32, 0, 0, 0);

    // device.lock().unwrap().cmd_execute_commands(primary_cb, &[grid_cb]);
    device.lock().unwrap().cmd_end_render_pass(primary_cb);
    device.lock().unwrap().end_command_buffer(primary_cb).unwrap();


    Ok((primary_cb))
}