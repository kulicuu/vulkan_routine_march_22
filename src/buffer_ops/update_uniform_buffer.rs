

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



#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}


pub unsafe fn update_uniform_buffer
(
    device: Arc<Mutex<DeviceLoader>>,
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
        device.lock().unwrap().map_memory(
            ubo_mems[current_image],
            0,
            buffer_size,
            vk::MemoryMapFlags::empty(),
            ).expect("Failed to map memory.") as *mut UniformBufferObject;
    data_ptr.copy_from_nonoverlapping(uni_transform_slice.as_ptr(), uni_transform_slice.len());
    device.lock().unwrap().unmap_memory(ubo_mems[current_image]);
    
}