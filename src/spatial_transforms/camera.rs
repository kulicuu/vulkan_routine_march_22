

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
#[derive(Debug, Clone, Copy)]
pub struct Attitude {
    // logically, the third axis normal can be derived from the other two, memoization indicates the third.
    pub roll_axis_normal: glm::Vec3,  // forward axis normal.
    pub pitch_axis_normal: glm::Vec3, // right axis normal
    pub yaw_axis_normal: glm::Vec3, // up axis normal
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub attitude: Attitude,
    pub position: glm::Vec3,
}


#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ControlInput {
    pub roll: i32,
    pub pitch: i32,
    pub yaw: i32,
    pub skew: i32,
}

pub fn transform_camera
<'a>
(
    camera: &mut Camera,
    view_matrix: &mut glm::Mat4,
    control_input: &mut ControlInput,
)
-> Result <(), &'a str>
{
    let scalar_45 = 0.03;
    // We are getting quantized packets of inputs, if they were big enough batches we'd ahve to interleave the operations to 
    // parody instantaneous parallel control inputs.  much yaw followed by much pitch is not the same as both applied simultaneously.
    camera.attitude.roll_axis_normal = glm::rotate_vec3(&camera.attitude.roll_axis_normal, (control_input.pitch as f32) 
        * scalar_45, &camera.attitude.pitch_axis_normal);
    camera.attitude.roll_axis_normal = glm::rotate_vec3(&camera.attitude.roll_axis_normal, (control_input.yaw as f32) 
        * scalar_45, &camera.attitude.yaw_axis_normal);
    camera.attitude.pitch_axis_normal = glm::rotate_vec3(&camera.attitude.pitch_axis_normal, (control_input.roll as f32) 
        * scalar_45, &camera.attitude.roll_axis_normal);
    camera.attitude.pitch_axis_normal = glm::rotate_vec3(&camera.attitude.pitch_axis_normal, (control_input.yaw as f32) 
        * scalar_45, &camera.attitude.yaw_axis_normal);
    camera.attitude.yaw_axis_normal = glm::rotate_vec3(&camera.attitude.yaw_axis_normal, (control_input.roll as f32) 
        * scalar_45, &camera.attitude.roll_axis_normal);
    camera.attitude.yaw_axis_normal = glm::rotate_vec3(&camera.attitude.yaw_axis_normal, (control_input.pitch as f32) 
        * scalar_45, &camera.attitude.pitch_axis_normal);

    *view_matrix = glm::look_at::<f32>
    (
        &camera.position,
        &(&camera.position + &camera.attitude.roll_axis_normal),
        &camera.attitude.yaw_axis_normal,
    );

    *control_input = ControlInput {
        roll: 0,
        pitch: 0,
        yaw: 0,
        skew: 0
    };

    Ok(())
}
