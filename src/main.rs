mod backup;
mod vulkan_6300;


use backup::*;
use vulkan_6300::*;



fn main() {

    // unsafe { vulkan_routine_2400() };  // safe in backup
    unsafe { vulkan_routine_6300() };
}
