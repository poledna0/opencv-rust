use opencv::core::{Rect, Size, Vector};
use opencv::prelude::*;
use opencv::{core, highgui, imgproc, videoio, Result};

opencv::opencv_branch_5! {
	use opencv::xobjdetect::{CascadeClassifier, CASCADE_SCALE_IMAGE};
}

opencv::not_opencv_branch_5! {
	use opencv::objdetect::{CascadeClassifier, CASCADE_SCALE_IMAGE};
}

fn main() -> Result<()> {
    const JANELA: &str = "camera";
    highgui::named_window_def(JANELA)?;
    let xml = core::find_file_def("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")?; // ou haarcascades/haarcascade_frontalface_alt.xml
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;

    let mut rosto = CascadeClassifier::new(&xml)?;

    loop {
        let mut frame = Mat::default();
        camera.read(&mut frame)?;

        let mut cinza = Mat::default();
        imgproc::cvt_color_def(&frame, &mut cinza, imgproc::COLOR_BGR2GRAY)?;

        let mut reduzida = Mat::default();
        imgproc::resize(&cinza, &mut reduzida, Size::new(0, 0), 0.25, 0.25, imgproc::INTER_LINEAR)?;

        let mut rostos = Vector::new();
        rosto.detect_multi_scale(
            &reduzida,
            &mut rostos,
            1.1,
            2,
            CASCADE_SCALE_IMAGE,
            Size::new(30, 30),
            Size::new(0, 0),
        )?;

        for r in rostos {
            //println!("Rosto: {r:?}");
            let cordenada_img_grande = Rect::new(r.x * 4, r.y * 4, r.width * 4, r.height * 4);
            imgproc::rectangle_def(&mut frame, cordenada_img_grande, (0, 255, 0).into())?;
        }

        highgui::imshow(JANELA, &frame)?;
        if highgui::wait_key(10)? > 0 {
            break;
        }
    }
    Ok(())
}