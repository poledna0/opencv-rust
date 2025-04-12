use opencv::{
    core,
    highgui,
    imgcodecs,
    prelude::*,
    Result,
};

fn main() -> Result<()> {
    let image = imgcodecs::imread("imagem.jpg", imgcodecs::IMREAD_COLOR)?;
    highgui::imshow("Janela", &image)?;
    highgui::wait_key(0)?;
    Ok(())
}
