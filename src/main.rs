mod vulkan_8700;
mod utilities;
mod buffer_ops;
mod data_structures;
mod spatial_transforms;
mod pipelines;
mod command_buffers;
mod threads;

mod precursors;

fn main() {
    unsafe { vulkan_8700::vulkan_routine_8700() };    
}
