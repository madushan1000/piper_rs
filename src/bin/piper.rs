use std::{collections::HashMap, fs::File, path::Path};

use burn::{
    Tensor,
    backend::{self, libtorch::LibTorchDevice},
    config::Config,
    tensor::Int,
};
use burn_backend::DType;
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};
use piper_rs::{
    print_tensor,
    utils::phonemes_to_ids,
    vits::{VitsConfig, VitsModel},
};

use clap::{Parser, ValueEnum};

type B = backend::LibTorch;
type Device = LibTorchDevice;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Args {
    Convert {
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        model_config: String,
        #[arg(long)]
        model_name: String,
        #[arg(long)]
        model_quality: Quality,
        #[arg(long)]
        output_path: String,
    },
    Run {
        #[arg(long)]
        model_config: String,
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        target_text: String,
        #[arg(long)]
        output_path: Option<String>,
        #[arg(long)]
        noise_scale: Option<f64>,
        #[arg(long)]
        length_scale: Option<f64>,
        #[arg(long)]
        noise_scale_w: Option<f64>,
        #[arg(long)]
        sid: Option<usize>,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum Quality {
    High,
    Medium,
    Low,
}

fn main() {
    match Args::parse() {
        Args::Convert {
            model_path,
            model_config,
            model_name,
            model_quality,
            output_path,
        } => convert(
            model_path,
            model_config,
            model_name,
            model_quality,
            output_path,
        )
        .unwrap(),
        run_args @ Args::Run { .. } => run(run_args),
    }
}

fn convert(
    model_path: String,
    model_config: String,
    model_name: String,
    model_quality: Quality,
    output_path: String,
) -> piper_rs::Result<()> {
    let model_config = File::open(model_config)?;
    let model_config: serde_json::Value = serde_json::from_reader(model_config)?;

    let phoneme_id_map = model_config["phoneme_id_map"].as_object().and_then(|map| {
        let mut id_map = HashMap::new();
        for (key, val) in map {
            let val = val.as_array().expect("phoneme id is not an array")[0]
                .as_i64()
                .expect("phoneme id is not an integer");
            id_map.insert(key.clone(), val);
        }
        Some(id_map)
    });

    let num_speakers = model_config["num_speakers"].as_u64().unwrap_or(1);
    let speaker_id_map = model_config["speaker_id_map"].as_object().and_then(|map| {
        let mut s_map = HashMap::new();
        for (key, val) in map {
            let val = val.as_i64().expect("speaker_id is not an integer");
            s_map.insert(key.clone(), val);
        }
        Some(s_map)
    });
    let espeak_voice = model_config["espeak"]["voice"].as_str().unwrap_or("en-us");

    let inference_noise_scale = model_config["inference"]["noise_scale"]
        .as_f64()
        .unwrap_or(0.667);
    let inference_length_scale = model_config["inference"]["length_scale"]
        .as_f64()
        .unwrap_or(1.0);
    let inference_noise_w = model_config["inference"]["noise_w"].as_f64().unwrap_or(0.8);

    let config = VitsConfig::new();
    let config = config
        .with_phoneme_id_map(phoneme_id_map)
        .with_num_speakers(num_speakers as usize)
        .with_espeak_voice(espeak_voice.to_string())
        .with_inference_noise_scale(inference_noise_scale)
        .with_inference_length_scale(inference_length_scale)
        .with_inference_noise_w(inference_noise_w)
        .with_speaker_id_map(speaker_id_map)
        .with_name(model_name);

    let config = match model_quality {
        Quality::High => config
            .with_resblock(1)
            .with_resblock_kernel_sizes(vec![3, 7, 11])
            .with_resblock_dilation_sizes(vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]])
            .with_upsample_rates(vec![8, 8, 2, 2])
            .with_upsample_initial_channel(512)
            .with_upsample_kernel_sizes(vec![16, 16, 4, 4])
            .with_quality("high".into()),
        Quality::Low => config
            .with_resblock(2)
            .with_resblock_kernel_sizes(vec![3, 5, 7])
            .with_resblock_dilation_sizes(vec![vec![1, 2], vec![2, 6], vec![3, 12]])
            .with_upsample_rates(vec![8, 8, 4])
            .with_upsample_initial_channel(256)
            .with_upsample_kernel_sizes(vec![16, 16, 8])
            .with_quality("low".into()),
        Quality::Medium => config,
    };

    let device: Device = Default::default();
    let mut model: VitsModel<B> = config.clone().init(&device);

    let model_path = Path::new(&model_path);

    let mut store = PytorchStore::from_file(model_path)
        .map_indices_contiguous(false)
        .allow_partial(true)
        .skip_enum_variants(true);

    let res = model.load_from(&mut store);

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

    let output_path = Path::new(&output_path);
    if !output_path.exists() {
        std::fs::create_dir_all(&output_path)?;
    }

    let output_name = format!(
        "{}_{}_{}",
        config.name,
        config.quality,
        model_path.file_name().unwrap().to_str().unwrap()
    );

    let mut store = BurnpackStore::from_file(output_path.join(&output_name).with_extension("bpk"))
        .overwrite(true);

    model.save_into(&mut store)?;

    config.save(output_path.join(output_name).with_extension("json"))?;

    Ok(())
}

fn run(run_args: Args) {
    let (
        model_config,
        model_path,
        target_text,
        output_path,
        noise_scale,
        length_scale,
        noise_scale_w,
        sid,
    ) = match run_args {
        Args::Run {
            model_config,
            model_path,
            target_text,
            output_path,
            noise_scale,
            length_scale,
            noise_scale_w,
            sid,
        } => (
            model_config,
            model_path,
            target_text,
            output_path,
            noise_scale,
            length_scale,
            noise_scale_w,
            sid,
        ),
        _ => panic!("shouldn't be here"),
    };

    let output_path = output_path.unwrap_or("output.wav".into());

    let device: Device = Default::default();

    let config = VitsConfig::load(model_config).unwrap();
    let mut vits_model: VitsModel<B> = config.clone().init(&device);
    let mut store = BurnpackStore::from_file(model_path).allow_partial(true);

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

    let speaker_id = sid;

    let noise_scale = noise_scale.unwrap_or(config.inference_noise_scale);
    let length_scale = length_scale.unwrap_or(config.inference_length_scale);
    let noise_scale_w = noise_scale_w.unwrap_or(config.inference_noise_w);

    let mut wav: Vec<f32> = vec![];

    let phonemes =
        &espeak_rs::text_to_phonemes(&target_text, &config.espeak_voice, None, false, false)
            .unwrap();
    for sent in phonemes {
        println!("phonemes: {sent}");
        let ipa = phonemes_to_ids(sent, config.phoneme_id_map.as_ref());

        let phoneme_lengths = ipa.len();
        let chunk = vits_model.forward(
            Tensor::<B, 1, Int>::from_ints(&ipa[..], &device).unsqueeze(),
            Tensor::from_ints([phoneme_lengths], &device),
            noise_scale,
            length_scale,
            noise_scale_w,
            speaker_id,
            None,
        );
        let chunk: Vec<f32> = chunk.cast(DType::F32).to_data().to_vec().unwrap();
        wav.extend(chunk);
    }

    //println!("{:?}", wav);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: vits_model.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec).unwrap();

    for s in wav.iter() {
        let i = ((*s * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16;
        writer.write_sample(i).unwrap();
    }
    writer.finalize().unwrap();
}
