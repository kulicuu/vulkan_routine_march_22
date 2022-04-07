


// PIpeline 102 is an experiment to draw a 3d grid using line list as primitive topology.






pub unsafe fn pipeline_102
<'a>
(
    device: &erupt::DeviceLoader,
    format: &vk::SurfaceFormatKHR,
    swapchain_image_extent: &vk::Extent2D,
)
-> Result<
    (
        vk::Pipeline,
        vk::PipelineLayout,
        vk::RenderPass,
        vk::ImageView,
    ),
    &'a str>
{

}