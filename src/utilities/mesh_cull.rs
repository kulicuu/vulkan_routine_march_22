pub fn mesh_cull_9945
<'a>
(
    mut indices: Vec<u32>,
)
-> Result <Vec<u32>, &'a str>
{
    // let mut jdx : usize = 0;
    // let mut cool : bool = true;
    // while cool {
    //     let start = jdx as usize;
    //     let end = (jdx + 3) as usize; 
    //     indices.drain(start..end);
    //     jdx += 12;
    //     if jdx > (indices.len()) {
    //         cool = false;
    //     }
    // }
    indices.drain(20000..);
    Ok(indices)
}
