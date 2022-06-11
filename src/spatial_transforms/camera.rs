

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