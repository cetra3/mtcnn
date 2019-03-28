use std::error::Error;

use tensorflow::{Graph, ImportGraphDefOptions, Session, SessionOptions, SessionRunArgs, Tensor};

use std::path::PathBuf;
use structopt::StructOpt;

use image::{Rgba, GenericImageView};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

#[derive(StructOpt)]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

#[derive(Copy, Clone, Debug)]
// Make it a bit nicer to work with the results, by adding a more explanatory struct
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub prob: f32,
}

// The line colour never changes, so make it a `const`
const LINE_COLOUR: Rgba<u8> = Rgba {
    data: [0, 255, 0, 0],
};

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();

    //First, we load up the graph as a byte array
    let model = include_bytes!("mtcnn.pb");

    //Then we create a tensorflow graph from the model
    let mut graph = Graph::new();
    graph.import_graph_def(&*model, &ImportGraphDefOptions::new())?;

    let input_image = image::open(&opt.input)?;

    let mut flattened: Vec<f32> = Vec::new();

    for (_x, _y, rgb) in input_image.pixels() {
        flattened.push(rgb[2] as f32);
        flattened.push(rgb[1] as f32);
        flattened.push(rgb[0] as f32);
    }

    //The `input` tensor expects BGR pixel data.
    let input = Tensor::new(&[input_image.height() as u64, input_image.width() as u64, 3])
        .with_values(&flattened)?;

    //Use input params from the existing module
    let min_size = Tensor::new(&[]).with_values(&[20f32])?;
    let thresholds = Tensor::new(&[3]).with_values(&[0.6f32, 0.7f32, 0.7f32])?;
    let factor = Tensor::new(&[]).with_values(&[0.709f32])?;

    let mut args = SessionRunArgs::new();

    //Load default parameters
    args.add_feed(&graph.operation_by_name_required("min_size")?, 0, &min_size);
    args.add_feed(&graph.operation_by_name_required("thresholds")?,0, &thresholds);
    args.add_feed(&graph.operation_by_name_required("factor")?, 0, &factor);

    //Load our input image
    args.add_feed(&graph.operation_by_name_required("input")?, 0, &input);

    //Request the following outputs after the session runs
    let bbox = args.request_fetch(&graph.operation_by_name_required("box")?, 0);
    let prob = args.request_fetch(&graph.operation_by_name_required("prob")?, 0);

    let mut session = Session::new(&SessionOptions::new(), &graph)?;

    session.run(&mut args)?;

    //Our bounding box extents
    let bbox_res: Tensor<f32> = args.fetch(bbox)?;
    //Our facial probability
    let prob_res: Tensor<f32> = args.fetch(prob)?;

    //Let's store the results as a Vec<BBox>
    let mut bboxes = Vec::new();

    let mut i = 0;
    let mut j = 0;

    //While we have responses, iterate through
    while i < bbox_res.len() {
        //Add in the 4 floats from the `bbox_res` array.
        //Notice the y1, x1, etc.. is ordered differently to our struct definition.
        bboxes.push(BBox {
            y1: bbox_res[i],
            x1: bbox_res[i + 1],
            y2: bbox_res[i + 2],
            x2: bbox_res[i + 3],
            prob: prob_res[j], // Add in the facial probability
        });

        //Step `i` ahead by 4.
        i += 4;
        //Step `i` ahead by 1.
        j += 1;
    }

    println!("BBox Length: {}, BBoxes:{:#?}", bboxes.len(), bboxes);

    //We don't want to change the input image, but create a new one based on it.
    let mut output_image = input_image.clone();

    //Iterate through all bounding boxes
    for bbox in bboxes {

        //Create a `Rect` from the bounding box.
        let rect = Rect::at(bbox.x1 as i32, bbox.y1 as i32)
            .of_size((bbox.x2 - bbox.x1) as u32, (bbox.y2 - bbox.y1) as u32);

        //Draw a green line around the bounding box
        draw_hollow_rect_mut(&mut output_image, rect, LINE_COLOUR);
    }

    //Once we've modified the image we save it in the output location.
    output_image.save(&opt.output)?;

    Ok(())
}
