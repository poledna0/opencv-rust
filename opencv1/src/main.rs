use opencv;

fn main() -> opencv::Result<()> {
    let imagem = opencv::imgcodecs::imread("imagem.png", opencv::imgcodecs::IMREAD_COLOR)?;
    opencv::highgui::imshow("Janela", &imagem)?;
    opencv::highgui::wait_key(0)?;
    Ok(())
}
