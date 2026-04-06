use std::error;
pub mod utils;
pub mod vits;

pub type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

#[macro_export]
macro_rules! print_tensor {
    ($data:expr, $device:expr, $($d:literal),+) => {
        match $data.shape.len() {
            $($d => {
                let t = Tensor::<burn::backend::LibTorch, $d>::from_data($data, $device);
                println!("{}", t);
            })*
            n => eprintln!("Unsupported rank: {}", n),
        }
    };
}
