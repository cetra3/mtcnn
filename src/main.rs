use crate::mtcnn::overlay;
use crate::mtcnn::BBox;
use image::DynamicImage;
use image::JPEG;
use std::error::Error;
use std::sync::Arc;

use structopt::StructOpt;

use std::env;

use actix_web::http::header::CONTENT_TYPE;
use actix_web::http::{HeaderValue, StatusCode};
use actix_web::{middleware, web, App, Error as ActixError, HttpResponse, HttpServer};
use futures::{Future, Stream};


use image::ImageError;

use log::info;

mod mtcnn;

use mtcnn::Mtcnn;

//Convenience type for our Mtcnn Arc we're gonna pass around
type WebMtcnn = web::Data<Arc<Mtcnn>>;

#[derive(StructOpt)]
struct Opt {
    #[structopt(
        short = "l",
        long = "listen",
        help = "Listen Address",
        default_value = "127.0.0.1:8000"
    )]
    listen: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    //Set the `RUST_LOG` var if none is provided
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "mtcnn=DEBUG,actix_web=INFO");
    }

    //Create a timestamped logger
    pretty_env_logger::init_timed();

    let opt = Opt::from_args();

    let mtcnn = Arc::new(Mtcnn::new()?);

    info!("Listening on: {}", opt.listen);

    Ok(HttpServer::new(move || {
        App::new()
            .data(mtcnn.clone()) //Add in our data handler
            .wrap(middleware::Logger::default()) //Add in a logger to see the requets coming through
            .service(web::resource("/api/v1/bboxes").to_async(return_bboxes))
            .service(web::resource("/api/v1/overlay").to_async(return_overlay))
    })
    .bind(&opt.listen)? // Use the listener from the command arguments
    .run()?)
}

// Allows us to get an image from a web payload
fn get_image(stream: web::Payload) -> impl Future<Item = DynamicImage, Error = ActixError> {
    stream
        .concat2()
        .from_err()
        .and_then(move |bytes| web::block(move || image::load_from_memory(&bytes)).from_err())
}

fn get_bboxes(
    img: DynamicImage,
    mtcnn: WebMtcnn,
) -> impl Future<Item = (DynamicImage, Vec<BBox>), Error = ActixError> {
    web::block(move || {
        mtcnn
            .run(&img)
            .map_err(|e| e.to_string())
            .map(|bboxes| (img, bboxes))
    })
    .from_err()
}

fn get_overlay(
    img: DynamicImage,
    bboxes: Vec<BBox>,
) -> impl Future<Item = Vec<u8>, Error = ActixError> {
    web::block(move || {
        let output_img = overlay(&img, &bboxes);

        let mut buffer = Vec::new();

        output_img.write_to(&mut buffer, JPEG)?;

        Ok(buffer) as Result<_, ImageError> // Type annotations required for the web::block
    })
    .from_err()
}

//This function purely returns a `json` representation of the bounding boxes
fn return_bboxes(
    stream: web::Payload,
    mtcnn: WebMtcnn,
) -> impl Future<Item = HttpResponse, Error = ActixError> {
    get_image(stream)
        .and_then(move |img| get_bboxes(img, mtcnn))
        .map(|(_img, bboxes)| HttpResponse::Ok().json(bboxes))
}

//This function returns our traditional overlay
fn return_overlay(
    stream: web::Payload,
    mtcnn: WebMtcnn,
) -> impl Future<Item = HttpResponse, Error = ActixError> {
    get_image(stream)
        .and_then(move |img| {
            get_bboxes(img, mtcnn)
        })
        .and_then(|(img, bbox) | get_overlay(img, bbox))
        .map(|buffer| {
            let mut response = HttpResponse::with_body(StatusCode::OK, buffer.into());
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("image/jpeg"));
            response
        })
}
