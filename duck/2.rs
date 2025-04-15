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

    let xml = core::find_file_def("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")?;
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !camera.is_opened()? {
        eprintln!("Erro ao acessar a câmera.");
        return Ok(());
    }

    let mut rosto = CascadeClassifier::new(&xml)?;

    let mut frame = Mat::default();
    let mut cinza = Mat::default();
    let mut reduzida = Mat::default();

    let mut count = 0;

    loop {
        count += 1;
        if count % 3 != 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue; // Processa apenas um em cada 3 frames
        }

        camera.read(&mut frame)?;


        imgproc::cvt_color_def(&frame, &mut cinza, imgproc::COLOR_BGR2GRAY)?;
        imgproc::resize(&cinza, &mut reduzida, Size::new(0, 0), 0.15, 0.15, imgproc::INTER_LINEAR)?; // Redução de escala agressiva

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
        rostos.shrink_to_fit(); // Minimiza memória alocada pelo vetor

        for r in &rostos {
            let cordenada_img_grande = Rect::new(r.x * 6, r.y * 6, r.width * 6, r.height * 6); // Ajusta multiplicador para refletir nova escala
            imgproc::rectangle_def(&mut frame, cordenada_img_grande, (0, 255, 0).into())?;
        }

        highgui::imshow(JANELA, &frame)?;

        if highgui::wait_key(50)? > 0 {
            break;
        }
    }

    Ok(())
}
