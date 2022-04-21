#![feature(drain_filter)]



mod vulkan_6300;
mod precursors;
mod pipeline_101;
mod pipeline_102;


use vulkan_6300::*;



fn main() {

    // unsafe { vulkan_routine_2400() };  // safe in backup
    unsafe { vulkan_routine_6300() };


    // let (mut vertices, mut indices) = load_model().unwrap();
    // println!("vertices.len(): {:?}, indices.len(): {:?}", vertices.len(), indices.len());
}
