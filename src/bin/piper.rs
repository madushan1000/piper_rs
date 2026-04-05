use std::any::type_name_of_val;

use burn::{
    Tensor,
    backend::{self, libtorch::LibTorchDevice},
    config::Config,
    prelude::*,
};
use burn_backend::DType;
use burn_store::{ModuleSnapshot, ModuleStore, PytorchStore};
use piper_rs::{print_tensor, vits::{VitsConfig, VitsModel, generate_path, sequence_mask}};

fn main() {
    type B = backend::LibTorch;
    let device: LibTorchDevice = Default::default();

    let config = VitsConfig::load("config.json").unwrap();
    let mut vits_model: VitsModel<B> = config.init(&device);
    //println!("{:?}", vits_model);
    let mut store = PytorchStore::from_file(
        "../checkpoints/en/en_US/ryan/medium/epoch=4641-step=3104302.contiguous.ckpt",
    )
    //.with_key_remapping(r"^model_g\.dp\.flows\.3", r"model_g.dp.flows.2")
    .map_indices_contiguous(false)
    .allow_partial(true)
    .skip_enum_variants(true);
    //.with_top_level_key("state_dict");
    let res = vits_model.load_from(&mut store);

    match res {
        Ok(val) => {
            println!("{}", val);
            println!("miss {:#?}", val.missing);
            println!("unused {:#?}", val.unused);
        }
        Err(val) => {
            println!("{}", val);
        }
    };

    //let mut bundel = PytorchStore::from_file("../piper1-gpl/bundle.pt");

    //let mut y_mask = Tensor::from_data(bundel.get_snapshot("y_mask").unwrap().unwrap().to_data().unwrap(), &device);
    //let mut z_p = Tensor::from_data(bundel.get_snapshot("z_p").unwrap().unwrap().to_data().unwrap(), &device);

    //println!("y_mask: {y_mask}");
    //println!("z_p: {z_p}");
    //let z = vits_model.model_g.flow.forward(z_p, y_mask.clone(), None, true);
    //println!("z: {z}");

    //let max_len = -1isize;
    //let o = vits_model.model_g
    //    .dec
    //    .forward((z * y_mask).slice([s![..], s![..], s![..=max_len]]), None);

    //println!("o: {o}");
    //(o, attn, y_mask, (z, z_p, m_p, logs_p))




    let text = "It's kind of remarkable how fucking stupid they all are";
    let phoneme_ids = [
        1, 0, 74, 0, 32, 0, 31, 0, 3, 0, 23, 0, 120, 0, 14, 0, 74, 0, 26, 0, 17, 0, 3, 0, 102, 0,
        34, 0, 3, 0, 88, 0, 128, 0, 25, 0, 120, 0, 51, 0, 122, 0, 88, 0, 23, 0, 59, 0, 15, 0, 59,
        0, 24, 0, 3, 0, 20, 0, 121, 0, 14, 0, 100, 0, 3, 0, 19, 0, 120, 0, 102, 0, 23, 0, 74, 0,
        44, 0, 3, 0, 31, 0, 32, 0, 120, 0, 33, 0, 122, 0, 28, 0, 74, 0, 17, 0, 3, 0, 41, 0, 18, 0,
        74, 0, 3, 0, 120, 0, 54, 0, 122, 0, 24, 0, 3, 0, 51, 0, 122, 0, 88, 0, 2,
    ];
    let phoneme_lengths = phoneme_ids.len();
    let noise_scale = 0.6670;
    let length_sacle = 1.0000;
    let noise_scale_w = 0.8000;
    let speaker_id = None;

    let wav = vits_model.forward(
        Tensor::from_ints([phoneme_ids], &device),
        Tensor::from_ints([phoneme_lengths], &device),
        noise_scale,
        length_sacle,
        noise_scale_w,
        speaker_id,
        None,
    );
    println!("{:?}", wav);
    let wav: Vec<f32> = wav.cast(DType::F32).to_data().to_vec().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 22050,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec).unwrap();

    for s in wav.iter() {
        let i = ((*s * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16;
        writer.write_sample(i).unwrap();
    }
    writer.finalize().unwrap();
}

//#[derive(Module, Debug)]
//struct Model<B: Backend> {
//    k: Option<Param<Tensor<B, 3>>>,
//    v: Option<Param<Tensor<B, 3>>>,
//}
//
//fn main() {
//    type B = backend::Cpu;
//    let device: CpuDevice = Default::default();
//
//    let mut model: Model<B> = Model {
//        k: Some(Param::from_tensor(Tensor::random(
//            [1, 9, 96],
//            Default::default(),
//            &device,
//        ))),
//        v: Some(Param::from_tensor(Tensor::random(
//            [1, 9, 96],
//            Default::default(),
//            &device,
//        ))),
//    };
//
//    let mut store = PytorchStore::from_file("s.pt");
//
//    println!("{:?}", model.load_from(&mut store));
