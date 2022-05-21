#![feature(drain_filter)]


mod vulkan_6400;
mod vulkan_6300;
mod precursors;
mod pipeline_101;
mod pipeline_102;


use vulkan_6400::*;
use vulkan_6300::*;


fn main() {

    unsafe { vulkan_routine_6400() };
    // unsafe { vulkan_routine_6300() };
    
}
