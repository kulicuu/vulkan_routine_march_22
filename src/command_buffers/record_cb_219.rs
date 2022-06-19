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
    command_pool: Arc<Mutex<vk::CommandPool>>,
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
    let mut dvc = device.lock().unwrap();
    let mut cpool = command_pool.lock().unwrap();
    dvc.reset_command_pool(*cpool, vk::CommandPoolResetFlags::empty()).unwrap();
    let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(*cpool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let primary_cb = dvc.allocate_command_buffers(&allocate_info).unwrap()[0];
    primary_command_buffers[image_index as usize] = primary_cb;
    let pri_cb_begin_info = vk::CommandBufferBeginInfoBuilder::new();
    dvc.begin_command_buffer(primary_cb, &pri_cb_begin_info).unwrap();
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
    dvc.cmd_begin_render_pass(
        primary_cb,
        &render_pass_begin_info,
        vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
    );
    dvc.cmd_bind_pipeline(primary_cb, vk::PipelineBindPoint::GRAPHICS, primary_pipeline);
    dvc.cmd_bind_index_buffer(primary_cb, ib, 0, vk::IndexType::UINT32);
    dvc.cmd_bind_vertex_buffers(primary_cb, 0, &[vb], &[0]);
    dvc.cmd_bind_descriptor_sets(primary_cb, vk::PipelineBindPoint::GRAPHICS, primary_pipeline_layout, 0, &d_sets, &[]);
    let ptr = std::ptr::addr_of!(pc_view) as *const c_void;
    dvc.cmd_push_constants
    (
        primary_cb,
        primary_pipeline_layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        // std::mem::size_of::<PushConstants>() as u32,
        std::mem::size_of::<glm::Mat4>() as u32,
        ptr,
    );
    dvc.cmd_draw_indexed(primary_cb, (indices_terr.len()) as u32, ((indices_terr.len()) / 3) as u32, 0, 0, 0);
    dvc.cmd_end_render_pass(primary_cb);
    dvc.end_command_buffer(primary_cb).unwrap();
    drop(dvc);
    drop(cpool);
    Ok((primary_cb))
}