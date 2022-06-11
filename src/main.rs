// #![feature(drain_filter)]


// mod vulkan_6400;
// mod vulkan_6300;
// mod vulkan_8300;
// mod vulkan_8400;
// mod vulkan_8500;
// mod vulkan_8600;
mod vulkan_8700;
mod utilities;
mod buffer_ops;

mod precursors;
mod pipeline_101;
mod pipeline_102;
// mod startup_vulkan;


use vulkan_8700::*;
// use vulkan_8600::*;
// use vulkan_8500::*;
// use vulkan_8400::*;
// use vulkan_8300::*;
// use vulkan_6400::*;
// use vulkan_6300::*;




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

use structopt::StructOpt;
const TITLE: &str = "vulkan-routine";


fn main() {
    unsafe { vulkan_routine_8700() }; 
    // unsafe { vulkan_routine_8600() }; 
    // unsafe { vulkan_routine_8500() };
    // unsafe { vulkan_routine_8400() };
    // unsafe { vulkan_routine_8300() };
    // unsafe { vulkan_routine_6400() };
    
    // unsafe { vulkan_routine_6300() };

    
}
