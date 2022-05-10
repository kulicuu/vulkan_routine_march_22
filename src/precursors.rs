

use erupt::{
    cstr,
    utils::{self, surface},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    vk::{Device, MemoryMapFlags},
};

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
    sync::Arc,
    thread,
    time,
};

pub unsafe fn create_precursors
<'a>
(
    instance: &InstanceLoader,
    surface: vk::SurfaceKHR,
)
-> Result<(
    vk::PhysicalDevice, 
    u32, 
    vk::SurfaceFormatKHR, 
    vk::PresentModeKHR, 
    vk::PhysicalDeviceProperties), &'a str>
{
    let device_extensions = vec![  // trying to type the contents of this vector to be able to pass between functions..todo.
        vk::KHR_SWAPCHAIN_EXTENSION_NAME,
        vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk::KHR_RAY_QUERY_EXTENSION_NAME,
        vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk::KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        vk::KHR_SPIRV_1_4_EXTENSION_NAME,
        vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    ];

    let (physical_device, queue_family, format, present_mode, device_properties) =

    instance.enumerate_physical_devices(None)
    .unwrap()
    .into_iter()
    .filter_map(|physical_device| {
        let queue_family = match instance
            .get_physical_device_queue_family_properties(physical_device, None)
            .into_iter()
            .enumerate()
            .position(|(i, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && instance
                        .get_physical_device_surface_support_khr(
                            physical_device,
                            i as u32,
                            surface,
                        )
                        .unwrap()
            }) {
            Some(queue_family) => queue_family as u32,
            None => return None,
        };
        let formats = instance
            .get_physical_device_surface_formats_khr(physical_device, surface, None)
            .unwrap();
        let format = match formats
            .iter()
            .find(|surface_format| {
                (surface_format.format == vk::Format::B8G8R8A8_SRGB
                    || surface_format.format == vk::Format::R8G8B8A8_SRGB)
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
            })
            .or_else(|| formats.get(0))
        {
            Some(surface_format) => *surface_format,
            None => return None,
        };
        let present_mode = instance
            .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
            .unwrap()
            .into_iter()
            .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
            .unwrap_or(vk::PresentModeKHR::FIFO_KHR);
        let supported_device_extensions = instance
            .enumerate_device_extension_properties(physical_device, None, None)
            .unwrap();
        let device_extensions_supported =
            device_extensions.iter().all(|device_extension| {
                let device_extension = CStr::from_ptr(*device_extension);

                supported_device_extensions.iter().any(|properties| {
                    CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                })
            });
        if !device_extensions_supported {
            return None;
        }
        let device_properties = instance.get_physical_device_properties(physical_device);
        Some((
            physical_device,
            queue_family,
            format,
            present_mode,
            device_properties,
        ))
    })
    .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 0,
    })
    .expect("No suitable physical device found");

    Ok((physical_device, queue_family, format, present_mode, device_properties))

}