use opencv::{
    highgui,
    imgcodecs,
    Result,
};

fn main() -> Result<()> {
    let imagem = imgcodecs::imread("iamgem.png", imgcodecs::IMREAD_COLOR)?;
    highgui::imshow("Janela", &imagem)?;
    highgui::wait_key(0)?;
    Ok(())
}
