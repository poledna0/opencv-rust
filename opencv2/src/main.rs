use opencv::{
    Result,
    prelude::*,
    videoio,
    highgui
};

fn main()-> Result<()>{

    let mut cam = videoio::VideoCapture::new(0,videoio::CAP_ANY)?;
    highgui::named_window("digitu", highgui::WINDOW_FULLSCREEN)?;

    let mut frame = Mat::default();
    loop {
        cam.read(&mut frame)?;

        highgui::imshow("digitu", &frame)?;

        let key = highgui::wait_key(10)?;

        if key == 113{
            break;
        }
    }

    Ok(())
}