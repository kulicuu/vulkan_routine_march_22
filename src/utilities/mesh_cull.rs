pub fn mesh_cull_9945
<'a>
(
    mut indices: Vec<u32>,
)
-> Result <Vec<u32>, &'a str>
{

    indices.drain(20000..);
    Ok(indices)
}
