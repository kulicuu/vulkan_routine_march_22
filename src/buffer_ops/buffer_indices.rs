
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


pub unsafe fn buffer_indices
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
    let mem_reqs = device.lock().unwrap().get_buffer_memory_requirements(ib);
    let alloc_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(1);
    let ib_mem = device.allocate_memory(&alloc_info, None).unwrap();
    device.bind_buffer_memory(ib, ib_mem, 0);
    let info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = device.lock().unwrap().allocate_command_buffers(&info).unwrap()[0];
    let info =  vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.lock().unwrap().begin_command_buffer(cb, &info).expect("Failed begin_command_buffer.");
    let info =  vk::BufferCopyBuilder::new()
        .src_offset(0)
        .dst_offset(0)
        .size(ib_size);
    device.cmd_copy_buffer(cb, sb, ib, &[info]);
    let cbs = &[cb];
    device.end_command_buffer(cb).expect("Failed to end command buffer.");
    let info = vk::SubmitInfoBuilder::new()
        .wait_semaphores(&[])
        .command_buffers(cbs)
        .signal_semaphores(&[]);
    device.queue_submit(queue, &[info], vk::Fence::null()).expect("Failed to queue submit.");
    Ok(ib)
}