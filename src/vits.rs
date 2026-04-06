use std::{any::type_name_of_val, collections::HashMap, f64::consts::PI};

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module, Param},
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, PaddingConfig1d, PaddingConfig2d,
        conv::{Conv1d, Conv1dConfig, ConvTranspose1dConfig},
    },
    prelude::*,
    tensor::{
        Bool, Int,
        activation::{gelu, leaky_relu, log_sigmoid, relu, sigmoid, softmax, softplus},
        module::{conv_transpose1d, conv1d, conv2d},
        ops::{ConvOptions, ConvTransposeOptions, PadMode, PaddedConvOptions},
    },
};
use burn_trace_shapes::trace_shapes;

#[derive(Config, Debug)]
pub struct VitsConfig {
    #[config(default = 32)]
    batch_size: usize,
    #[config(default = 22050)]
    sample_rate: usize,
    #[config(default = 256)]
    num_symbols: usize,
    #[config(default = 1)]
    num_speakers: usize,
    // audio
    #[config(default = 2)]
    resblock: usize,
    #[config(default = "vec![3, 5, 7]")]
    resblock_kernel_sizes: Vec<usize>,
    #[config(default = "vec![vec![1, 2], vec![2, 6], vec![3, 12]]")]
    resblock_dilation_sizes: Vec<Vec<usize>>,
    #[config(default = "vec![8, 8, 4]")]
    upsample_rates: Vec<usize>,
    #[config(default = 256)]
    upsample_initial_channel: usize,
    #[config(default = "vec![16, 16, 8]")]
    upsample_kernel_sizes: Vec<usize>,
    // mel
    #[config(default = 1024)]
    filter_length: usize,
    #[config(default = 256)]
    hop_length: usize,
    #[config(default = 1024)]
    win_length: usize,
    #[config(default = 80)]
    mel_channels: usize,
    #[config(default = 0.0)]
    mel_fmin: f32,
    #[config(default = "None")]
    mel_fmax: Option<f32>,
    //model
    #[config(default = 192)]
    inter_channels: usize,
    #[config(default = 192)]
    hidden_channels: usize,
    #[config(default = 768)]
    filter_channels: usize,
    #[config(default = 2)]
    n_heads: usize,
    #[config(default = 6)]
    n_layers: usize,
    #[config(default = 3)]
    kernel_size: usize,
    #[config(default = 0.1)]
    p_dropout: f64,
    #[config(default = 3)]
    n_layers_q: usize,
    #[config(default = false)]
    use_spectral_norm: bool,
    #[config(default = 0)]
    gin_channels: usize,
    #[config(default = true)]
    use_sdp: bool,
    #[config(default = 8192)]
    segment_size: usize,
    // training
    #[config(default = 2e-4)]
    learning_rate: f32,
    #[config(default = 1e-4)]
    learning_rate_d: f32,
    #[config(default = "(0.8, 0.99)")]
    betas: (f32, f32),
    #[config(default = "(0.5, 0.9)")]
    betas_d: (f32, f32),
    #[config(default = 1e-9)]
    eps: f32,
    #[config(default = 0.999875)]
    lr_decay: f32,
    #[config(default = 0.9999)]
    lr_decay_d: f32,
    #[config(default = 1.0)]
    init_lr_ratio: f32,
    #[config(default = 0)]
    warmup_epochs: usize,
    #[config(default = 45)]
    c_mel: usize,
    #[config(default = 1.0)]
    c_kl: f32,
    #[config(default = "None")]
    grad_clip: Option<f32>,
    #[config(default = "None")]
    vocoder_warmstart_ckpt: Option<String>,
    // unused
    #[config(default = "None")]
    dataset: Option<()>,
    #[config(default = "None")]
    pub phoneme_id_map: Option<HashMap<String, i64>>,
    #[config(default = "\"piper\".into()")]
    pub name: String,
    #[config(default = "\"medium\".into()")]
    pub quality: String,
    #[config(default = "\"en-us\".into()")]
    pub espeak_voice: String,
    #[config(default = 0.667)]
    pub inference_noise_scale: f64,
    #[config(default = 1.0)]
    pub inference_length_scale: f64,
    #[config(default = 0.8)]
    pub inference_noise_w: f64,
    #[config(default = "None")]
    pub speaker_id_map: Option<HashMap<String, i64>>
}

impl VitsConfig {
    pub fn init<B: Backend>(mut self, device: &B::Device) -> VitsModel<B> {
        assert_eq!(
            self.upsample_rates.iter().fold(1, |acc, x| acc * x),
            self.hop_length,
            "Upsample rates do not match hop length"
        );
        if self.num_speakers > 1 && self.gin_channels <= 0 {
            self.gin_channels = 512;
        }
        VitsModel {
            model_g: SynthesizerTrnCofnig::new(
                self.num_symbols,
                self.filter_length / 2 + 1,
                self.segment_size / self.hop_length,
                self.inter_channels,
                self.hidden_channels,
                self.filter_channels,
                self.n_heads,
                self.n_layers,
                self.kernel_size,
                self.p_dropout,
                self.resblock,
                self.resblock_kernel_sizes,
                self.resblock_dilation_sizes,
                self.upsample_rates,
                self.upsample_initial_channel,
                self.upsample_kernel_sizes,
            )
            .with_n_speakers(self.num_speakers)
            .with_gin_channels(self.gin_channels)
            .with_use_sdp(self.use_sdp)
            .init(device),
            model_d: MultiPeriodDiscriminatorConfig::new().init(device),
            sample_rate: self.sample_rate,
        }
    }
}

#[derive(Module, Debug)]
pub struct VitsModel<B: Backend> {
    pub model_g: SynthesizerTrn<B>,
    pub model_d: MultiPeriodDiscriminator<B>,
    pub sample_rate: usize,
}

impl<B: Backend> VitsModel<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        phoname_ids: Tensor<B, 2, Int>,
        phoname_lengths: Tensor<B, 1, Int>,
        noise_scale: f64,
        length_sacle: f64,
        noise_scale_w: f64,
        speaker_id: Option<usize>,
        max_len: Option<usize>,
    ) -> Tensor<B, 3> {
        self.model_g.infer(
            phoname_ids,
            phoname_lengths,
            noise_scale,
            length_sacle,
            noise_scale_w,
            speaker_id,
            max_len,
        )
    }
}

#[derive(Config, Debug)]
pub struct SynthesizerTrnCofnig {
    n_vocab: usize,
    spec_channels: usize,
    segment_size: usize,
    inter_channels: usize,
    hidden_channels: usize,
    filter_channels: usize,
    n_heads: usize,
    n_layers: usize,
    kernel_size: usize,
    p_dropout: f64,
    resblock: usize,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilation_sizes: Vec<Vec<usize>>,
    upsample_rates: Vec<usize>,
    upsample_initial_channel: usize,
    upsample_kernel_sizes: Vec<usize>,
    #[config(default = 1)]
    n_speakers: usize,
    #[config(default = 0)]
    gin_channels: usize,
    #[config(default = true)]
    use_sdp: bool,
}
impl SynthesizerTrnCofnig {
    fn init<B: Backend>(&self, device: &B::Device) -> SynthesizerTrn<B> {
        SynthesizerTrn {
            enc_p: TextEncoderConfig::new(
                self.n_vocab,
                self.inter_channels,
                self.hidden_channels,
                self.filter_channels,
                self.n_heads,
                self.n_layers,
                self.kernel_size,
                self.p_dropout,
            )
            .init(device),
            dec: GeneratorConfig::new(
                self.inter_channels,
                self.resblock,
                self.resblock_kernel_sizes.clone(),
                self.resblock_dilation_sizes.clone(),
                self.upsample_rates.clone(),
                self.upsample_initial_channel,
                self.upsample_kernel_sizes.clone(),
                self.gin_channels,
            )
            .init(device),
            enc_q: PosteriorEncoderConfig::new(
                self.spec_channels,
                self.inter_channels,
                self.hidden_channels,
                5,
                1,
                16,
                self.gin_channels,
            )
            .init(device),
            flow: ResidualCouplingBlockConfig::new(
                self.inter_channels,
                self.hidden_channels,
                5,
                1,
                4,
                self.gin_channels,
            )
            .init(device),
            dp: if self.use_sdp {
                SynthesizerTrnLayers::StochasticDurationPredictor(
                    StochasticDurationPredictorConfig::new(
                        self.hidden_channels,
                        192,
                        3,
                        0.5,
                        4,
                        self.gin_channels,
                    )
                    .init(device),
                )
            } else {
                SynthesizerTrnLayers::DurationPredictor(
                    DurationPredictorConfig::new(
                        self.hidden_channels,
                        256,
                        3,
                        0.5,
                        self.gin_channels,
                    )
                    .init(device),
                )
            },
            emb_g: if self.n_speakers > 1 {
                Some(EmbeddingConfig::new(self.n_speakers, self.gin_channels).init(device))
            } else {
                None
            },
            n_speakers: self.n_speakers,
        }
    }
}

#[derive(Module, Debug)]
pub struct SynthesizerTrn<B: Backend> {
    pub enc_p: TextEncoder<B>,
    pub dec: Generator<B>,
    pub enc_q: PosteriorEncoder<B>,
    pub flow: ResidualCouplingBlock<B>,
    pub dp: SynthesizerTrnLayers<B>,
    pub emb_g: Option<Embedding<B>>,
    pub n_speakers: usize,
}

impl<B: Backend> SynthesizerTrn<B> {
    fn infer(
        &self,
        x: Tensor<B, 2, Int>,
        x_lengths: Tensor<B, 1, Int>,
        noise_scale: f64,
        length_sacle: f64,
        noise_scale_w: f64,
        sid: Option<usize>,
        max_len: Option<usize>,
    ) -> Tensor<B, 3> {
        let (x, m_p, logs_p, x_mask) = self.enc_p.forward(x, x_lengths);
        let g = if self.n_speakers > 1 {
            match (self.emb_g.as_ref(), sid) {
                (Some(emb_g), Some(sid)) => {
                    Some(emb_g.forward(Tensor::from_ints([sid], &x.device())))
                }
                _ => panic!("Missing speaker id"),
            }
        } else {
            None
        };

        let logw = self
            .dp
            .forward(x, x_mask.clone(), None, g.clone(), true, noise_scale_w);

        let w = logw.exp() * x_mask.clone() * length_sacle;
        let w_ceil = w.ceil();
        let y_lengths = w_ceil.clone().sum_dims_squeeze(&[1, 2]).clamp_min(1).int();
        let y_mask = sequence_mask(
            y_lengths.clone(),
            Some(y_lengths.max().to_data().to_vec::<i64>().unwrap()[0] as usize),
        )
        .unsqueeze_dim(1)
        .float();
        let attn_mask = x_mask.unsqueeze_dim(2) * y_mask.clone().unsqueeze_dims(&[-1]);
        let attn = generate_path(w_ceil, attn_mask);

        let m_p = attn
            .clone()
            .squeeze_dim(1)
            .matmul(m_p.swap_dims(1, 2))
            .swap_dims(1, 2);

        let logs_p = attn
            .squeeze_dim(1)
            .matmul(logs_p.swap_dims(1, 2))
            .swap_dims(1, 2);

        let z_p =
            m_p.clone() + m_p.clone().random_like(Default::default()) * logs_p.exp() * noise_scale;
        let z = self.flow.forward(z_p, y_mask.clone(), g.clone(), true);

        let max_len = match max_len {
            Some(val) => val as isize,
            None => -1isize,
        };
        let o = self
            .dec
            .forward((z * y_mask).slice([s![..], s![..], s![..=max_len]]), g);
        //(o, attn, y_mask, (z, z_p, m_p, logs_p))
        o
    }
}

pub fn generate_path<B: Backend>(duration: Tensor<B, 3>, mask: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, _, t_y, t_x] = mask.dims();
    let cum_duration = duration.clone().cumsum(duration.dims().len() - 1);

    let cum_duration_flat = cum_duration.reshape([b * t_x]);
    let path = sequence_mask(cum_duration_flat.int(), Some(t_y)).float();
    let path = path.reshape([b, t_x, t_y]);
    let path = path.clone()
        - path
            .pad([(0, 0), (1, 0), (0, 0)], PadMode::Constant(0.0))
            .slice([s![..], s![..-1]]);
    path.unsqueeze_dim(1).swap_dims(2, 3) * mask
}

#[derive(Module, Debug)]
pub enum SynthesizerTrnLayers<B: Backend> {
    StochasticDurationPredictor(StochasticDurationPredictor<B>),
    DurationPredictor(DurationPredictor<B>),
}
impl<B: Backend> SynthesizerTrnLayers<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        w: Option<Tensor<B, 3>>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
        noise_scale: f64,
    ) -> Tensor<B, 3> {
        match self {
            Self::StochasticDurationPredictor(sdp) => {
                sdp.forward(x, x_mask, w, g, reverse, noise_scale)
            }
            Self::DurationPredictor(dp) => dp.forward(x, x_mask, g),
        }
    }
}

#[derive(Config, Debug)]
pub struct TextEncoderConfig {
    n_vocab: usize,
    out_channels: usize,
    hidden_channels: usize,
    filter_channels: usize,
    n_heads: usize,
    n_layers: usize,
    kernel_size: usize,
    p_dropout: f64,
}

impl TextEncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TextEncoder<B> {
        TextEncoder {
            emb: EmbeddingConfig::new(self.n_vocab, self.hidden_channels).init(device),
            encoder: EncoderConfig::new(
                self.hidden_channels,
                self.filter_channels,
                self.n_heads,
                self.n_layers,
                self.kernel_size,
                self.p_dropout,
            )
            .init(device),
            proj: Conv1dConfig::new(self.hidden_channels, self.out_channels * 2, 1).init(device),
            hidden_channels: self.hidden_channels,
            out_channels: self.out_channels,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    emb: Embedding<B>,
    encoder: Encoder<B>,
    proj: Conv1d<B>,
    hidden_channels: usize,
    out_channels: usize,
}

impl<B: Backend> TextEncoder<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        x_lengths: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let x = self.emb.forward(x) * (self.hidden_channels as f64).sqrt();

        let x = x.swap_dims(1, -1);

        let x_mask = sequence_mask(x_lengths, Some(x.shape()[2]))
            .unsqueeze_dim(1)
            .float();

        let x = self.encoder.forward(x * x_mask.clone(), x_mask.clone());
        let stats = self.proj.forward(x.clone()) * x_mask.clone();

        let split = stats.split(self.out_channels, 1);
        let m = split[0].clone();
        let logs = split[1].clone();

        (x, m, logs, x_mask)
    }
}

pub fn sequence_mask<B: Backend>(
    length: Tensor<B, 1, Int>,
    max_length: Option<usize>,
) -> Tensor<B, 2, Bool> {
    let max_length = match max_length {
        Some(val) => val,
        None => length.clone().max().into_data().to_vec::<u32>().unwrap()[0] as usize,
    };
    let x: Tensor<B, 1, Int> = Tensor::arange(0..max_length as i64, &length.device());
    //x.lower(length).int()
    x.unsqueeze_dim(0).lower(length.unsqueeze_dim(1))
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    hidden_channels: usize,
    filter_channels: usize,
    n_heads: usize,
    n_layers: usize,
    kernel_size: usize,
    p_dropout: f64,
    #[config(default = 4)]
    window_size: usize,
}
impl EncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        let mut attn_layers = vec![];
        let mut norm_layers_1 = vec![];
        let mut ffn_layers = vec![];
        let mut norm_layers_2 = vec![];

        for _ in 0..self.n_layers {
            attn_layers.push(
                MultiHeadAttentionConfig::new(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.n_heads,
                )
                .with_p_dropout(self.p_dropout)
                .with_window_size(Some(self.window_size))
                .init(device),
            );
            norm_layers_1.push(LayerNormConfig::new(self.hidden_channels).init(device));
            ffn_layers.push(
                FFNConfig::new(
                    self.hidden_channels,
                    self.hidden_channels,
                    self.filter_channels,
                    self.kernel_size,
                )
                .with_p_dropout(self.p_dropout)
                .init(device),
            );
            norm_layers_2.push(LayerNormConfig::new(self.hidden_channels).init(device));
        }
        Encoder {
            drop: DropoutConfig::new(self.p_dropout).init(),
            attn_layers,
            norm_layers_1,
            ffn_layers,
            norm_layers_2,
        }
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    drop: Dropout,
    attn_layers: Vec<MultiHeadAttention<B>>,
    norm_layers_1: Vec<LayerNorm<B>>,
    ffn_layers: Vec<FFN<B>>,
    norm_layers_2: Vec<LayerNorm<B>>,
}
impl<B: Backend> Encoder<B> {
    #[trace_shapes]
    fn forward(&self, x: Tensor<B, 3>, x_mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_mask = x_mask.clone().unsqueeze_dim(2) * x_mask.clone().unsqueeze_dims(&[-1]);
        let mut x = x * x_mask.clone();
        for i in 0..self.attn_layers.len() {
            let y = self.attn_layers[i].forward(x.clone(), x.clone(), Some(attn_mask.clone()));
            let y = self.drop.forward(y);
            //TODO check if swap_dims is correct here
            x = self.norm_layers_1[i]
                .forward((x + y).swap_dims(1, 2))
                .swap_dims(2, 1);

            let y = self.ffn_layers[i].forward(x.clone(), x_mask.clone());
            let y = self.drop.forward(y);
            //TODO check if swap_dims is correct here
            x = self.norm_layers_2[i]
                .forward((x + y).swap_dims(1, 2))
                .swap_dims(2, 1);
        }
        x * x_mask
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    channels: usize,
    out_channels: usize,
    n_heads: usize,
    #[config(default = 0.0)]
    p_dropout: f64,
    #[config(default = "None")]
    window_size: Option<usize>,
    #[config(default = true)]
    heads_share: bool,
    #[config(default = "None")]
    block_length: Option<usize>,
    #[config(default = false)]
    proximal_bias: bool,
    #[config(default = false)]
    proximal_init: bool,
}
impl MultiHeadAttentionConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> MultiHeadAttention<B> {
        let k_channels = self.channels / self.n_heads;
        let mut emb_rel_k = None;
        let mut emb_rel_v = None;
        if let Some(window_size) = self.window_size {
            let n_heads_rel = if self.heads_share { 1 } else { self.n_heads };
            let rel_stddev = (k_channels as f64).powf(-0.5);
            emb_rel_k = Some(Param::from_tensor(
                Tensor::random(
                    [n_heads_rel, window_size * 2 + 1, k_channels],
                    Default::default(),
                    device,
                ) * rel_stddev,
            ));
            emb_rel_v = Some(Param::from_tensor(
                Tensor::random(
                    [n_heads_rel, window_size * 2 + 1, k_channels],
                    Default::default(),
                    device,
                ) * rel_stddev,
            ));
        }

        let conv_q = Conv1dConfig::new(self.channels, self.channels, 1)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        let mut conv_k = Conv1dConfig::new(self.channels, self.channels, 1)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 })
            .init(device);

        if self.proximal_init {
            conv_k.weight = conv_q.weight.clone();
            conv_k.bias = conv_q.bias.clone();
        }

        MultiHeadAttention {
            conv_q,
            conv_k,
            conv_v: Conv1dConfig::new(self.channels, self.channels, 1)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
            conv_o: Conv1dConfig::new(self.channels, self.out_channels, 1).init(device),
            drop: DropoutConfig::new(self.p_dropout).init(),
            emb_rel_k,
            emb_rel_v,
            n_heads: self.n_heads,
            k_channels,
            window_size: self.window_size,
            proximal_bias: self.proximal_bias,
            block_length: self.block_length,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    conv_q: Conv1d<B>,
    conv_k: Conv1d<B>,
    conv_v: Conv1d<B>,
    conv_o: Conv1d<B>,
    drop: Dropout,
    emb_rel_k: Option<Param<Tensor<B, 3>>>,
    emb_rel_v: Option<Param<Tensor<B, 3>>>,
    n_heads: usize,
    k_channels: usize,
    window_size: Option<usize>,
    proximal_bias: bool,
    block_length: Option<usize>,
}

impl<B: Backend> MultiHeadAttention<B> {
    //'input': [(1, 192, 123), (1, 192, 123), (1, 1, 123, 123)], 'output': (1, 192, 123)
    #[trace_shapes]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        c: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let q = self.conv_q.forward(x);
        let k = self.conv_k.forward(c.clone());
        let v = self.conv_v.forward(c);

        let (x, _attn) = self.attention(q, k, v, attn_mask);

        self.conv_o.forward(x)
    }

    fn attention(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let [b, d, t_s] = key.dims();
        let t_t = query.dims()[2];
        let query = query
            .reshape([b, self.n_heads, self.k_channels, t_t])
            .swap_dims(2, 3);
        let key = key
            .reshape([b, self.n_heads, self.k_channels, t_s])
            .swap_dims(2, 3);
        let value = value
            .reshape([b, self.n_heads, self.k_channels, t_s])
            .swap_dims(2, 3);

        let mut scores =
            (query.clone() / (self.k_channels as f64).sqrt()).matmul(key.swap_dims(-2, -1));
        if self.window_size.is_some() {
            assert!(
                (t_s == t_t),
                "Relative attention is only available for self-attention."
            );
            let key_relative_embeddings =
                self._get_relative_embeddings(self.emb_rel_k.as_ref().unwrap().val(), t_s);

            let rel_logits = self._matmul_with_relative_keys(
                query / (self.k_channels as f64).sqrt(),
                key_relative_embeddings,
            );

            let scores_local = self._relative_position_to_absolute_position(rel_logits);

            scores = scores + scores_local;
        }

        if self.proximal_bias {
            assert!(
                t_s == t_t,
                "Proximal bias is only available for self-attention."
            );
            scores = scores.clone() + self._attention_bias_proximal(t_s, &scores.device())
        }
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask.equal_elem(0), -1e4);
            if let Some(block_length) = self.block_length {
                assert!(
                    t_s == t_t,
                    "Local attention is only available for self-attention."
                );
                let block_mask = Tensor::ones_like(&scores)
                    .triu(-(block_length as i64))
                    .tril(block_length as i64);
                scores = scores.mask_fill(block_mask.equal_elem(0), -1e4);
            }
        }

        let p_attn = softmax(scores.clone(), scores.dims().len() - 1);
        let p_attn = self.drop.forward(p_attn);
        let mut output = p_attn.clone().matmul(value);

        if self.window_size.is_some() {
            let reletive_weights = self._absolute_position_to_relative_position(p_attn.clone());
            let value_relative_embeddings =
                self._get_relative_embeddings(self.emb_rel_v.as_ref().unwrap().val(), t_s);
            output = output
                + self._matmul_with_relative_values(reletive_weights, value_relative_embeddings);
        }

        let output = output.swap_dims(2, 3).reshape([b, d, t_t]);

        return (output, p_attn);
    }

    fn _matmul_with_relative_keys(&self, x: Tensor<B, 4>, y: Tensor<B, 3>) -> Tensor<B, 4> {
        x.matmul(y.unsqueeze_dim(0).swap_dims(-2, -1))
    }

    fn _matmul_with_relative_values(&self, x: Tensor<B, 4>, y: Tensor<B, 3>) -> Tensor<B, 4> {
        x.matmul(y.unsqueeze_dim(0))
    }

    fn _attention_bias_proximal(&self, length: usize, device: &B::Device) -> Tensor<B, 4> {
        let r = Tensor::arange(0..length as i64, device);
        let diff: Tensor<B, 2, Int> = r.clone().unsqueeze_dim(0).sub(r.unsqueeze_dim(1));
        diff.abs()
            .float()
            .log1p()
            .neg()
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
    }

    fn _get_relative_embeddings(
        &self,
        relative_embeddings: Tensor<B, 3>,
        length: usize,
    ) -> Tensor<B, 3> {
        let window_size = self.window_size.unwrap_or(0);
        let pad_length = 0.max(length as isize - (window_size as isize + 1)) as usize;
        let slice_start_position = 0.max(window_size as isize + 1 - length as isize) as usize;
        let slice_end_position = (slice_start_position as isize + 2 * length as isize - 1) as usize;

        let padded_relative_embeddings = if pad_length > 0 {
            relative_embeddings.pad(
                [(0, 0), (pad_length, pad_length), (0, 0)],
                PadMode::Constant(0.0),
            )
        } else {
            relative_embeddings
        };

        let used_relative_embeddings = padded_relative_embeddings
            .slice([s![..], s![slice_start_position..slice_end_position]]);
        used_relative_embeddings
    }

    fn _absolute_position_to_relative_position(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, length, _] = x.dims();
        let x = x.pad(
            [(0, 0), (0, 0), (0, 0), (0, length - 1)],
            PadMode::Constant(0.0),
        );
        let x_flat = x.reshape([batch, heads, (length * length) + (length * (length - 1))]);
        let x_flat = x_flat.pad([(0, 0), (0, 0), (length, 0)], PadMode::Constant(0.0));
        let x_final = x_flat.reshape([batch, heads, length, 2 * length]).slice([
            s![..],
            s![..],
            s![..],
            s![1..],
        ]);
        x_final
    }

    fn _relative_position_to_absolute_position(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, length, _] = x.dims();
        let x = x.pad([(0, 0), (0, 0), (0, 0), (0, 1)], PadMode::Constant(0.0));
        let x_flat = x.reshape([batch, heads, length * 2 * length]);
        let x_flat = x_flat.pad([(0, 0), (0, 0), (0, length - 1)], PadMode::Constant(0.0));

        let x_final = x_flat
            .reshape([batch, heads, length + 1, (2 * length) - 1])
            .slice([s![..], s![..], s![..length], s![length - 1..]]);
        x_final
    }
}

#[derive(Config, Debug)]
pub struct FFNConfig {
    in_channels: usize,
    out_channels: usize,
    filter_channels: usize,
    kernel_size: usize,
    #[config(default = 0.0)]
    p_dropout: f64,
    #[config(default = "\"\".to_string()")]
    activation: String,
    #[config(default = false)]
    causal: bool,
}
impl FFNConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> FFN<B> {
        FFN {
            conv_1: Conv1dConfig::new(self.in_channels, self.filter_channels, self.kernel_size)
                .init(device),
            conv_2: Conv1dConfig::new(self.filter_channels, self.out_channels, self.kernel_size)
                .init(device),
            drop: DropoutConfig::new(self.p_dropout).init(),
            causal: self.causal,
            activation: self.activation.clone(),
            kernel_size: self.kernel_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct FFN<B: Backend> {
    conv_1: Conv1d<B>,
    conv_2: Conv1d<B>,
    drop: Dropout,
    causal: bool,
    activation: String,
    kernel_size: usize,
}

impl<B: Backend> FFN<B> {
    //'input': [(1, 192, 123), (1, 1, 123)], 'output': (1, 192, 123)
    #[trace_shapes]
    pub fn forward(&self, x: Tensor<B, 3>, x_mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let padding1 = if self.causal {
            self._causal_padding(x * x_mask.clone())
        } else {
            self._same_padding(x * x_mask.clone())
        };

        let x = self.conv_1.forward(padding1);

        let x = if self.activation == "gelu" {
            x.clone() * sigmoid(1.702 * x)
        } else {
            relu(x)
        };

        let x = self.drop.forward(x);

        let padding2 = if self.causal {
            self._causal_padding(x * x_mask.clone())
        } else {
            self._same_padding(x * x_mask.clone())
        };

        let x = self.conv_2.forward(padding2);

        x * x_mask
    }

    fn _causal_padding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.kernel_size == 1 {
            return x;
        }
        let pad_l = self.kernel_size - 1;
        x.pad([(0, 0), (0, 0), (pad_l, 0)], PadMode::Constant(0.0))
    }

    fn _same_padding(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.kernel_size == 1 {
            return x;
        }
        let pad_l = (self.kernel_size - 1) / 2;
        let pad_r = self.kernel_size / 2;
        x.pad([(0, 0), (0, 0), (pad_l, pad_r)], PadMode::Constant(0.0))
    }
}

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    initial_channel: usize,
    resblock: usize,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilation_sizes: Vec<Vec<usize>>,
    upsample_rates: Vec<usize>,
    upsample_initial_channel: usize,
    upsample_kernel_sizes: Vec<usize>,
    gin_channels: usize,
}
impl GeneratorConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        let mut ups = vec![];
        for (i, (u, k)) in self
            .upsample_rates
            .iter()
            .zip(self.upsample_kernel_sizes.iter())
            .enumerate()
        {
            ups.push(
                WNConvTranspose1dConfig::new(
                    self.upsample_initial_channel / (2_usize.pow(i as u32)),
                    self.upsample_initial_channel / (2_usize.pow(i as u32 + 1)),
                    k.clone(),
                    *u,
                )
                .with_padding((k - u) / 2)
                .init(device),
            );
        }

        let mut resblocks = vec![];
        let mut ch = 0;
        for i in 0..ups.len() {
            ch = self.upsample_initial_channel / (2_usize.pow(i as u32 + 1));
            for (j, (k, d)) in self
                .resblock_kernel_sizes
                .iter()
                .zip(self.resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(ResBlockConfig::new(self.resblock, ch, *k, d.clone()).init(device));
            }
        }

        Generator {
            conv_pre: Conv1dConfig::new(self.initial_channel, self.upsample_initial_channel, 7)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            ups,
            resblocks,
            conv_post: Conv1dConfig::new(ch, 1, 7)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .with_bias(false)
                .init(device),
            cond: if self.gin_channels != 0 {
                Some(
                    Conv1dConfig::new(self.gin_channels, self.upsample_initial_channel, 0)
                        .init(device),
                )
            } else {
                None
            },
            num_kernels: self.resblock_kernel_sizes.len(),
            lrelu_slope: 0.1,
        }
    }
}

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    conv_pre: Conv1d<B>,
    ups: Vec<WNConvTranspose1d<B>>,
    resblocks: Vec<ResBlock<B>>,
    conv_post: Conv1d<B>,
    cond: Option<Conv1d<B>>,
    num_kernels: usize,
    lrelu_slope: f64,
}

impl<B: Backend> Generator<B> {
    #[trace_shapes]
    pub fn forward(&self, x: Tensor<B, 3>, g: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let x = self.conv_pre.forward(x);
        let mut x = match (g, self.cond.as_ref()) {
            (Some(g), Some(cond)) => x + cond.forward(g),
            _ => x,
        };

        for (i, up) in self.ups.iter().enumerate() {
            x = leaky_relu(x, self.lrelu_slope);
            x = up.forward(x);
            let mut xs: Tensor<B, 3> = Tensor::zeros([1, 1, 1], &x.device());
            for (j, resblock) in self.resblocks.iter().enumerate() {
                let index: isize = j as isize - (i as isize * self.num_kernels as isize);
                if index == 0 {
                    xs = resblock.forward(x.clone(), None);
                } else if index > 0 && (index < self.num_kernels as isize) {
                    xs = xs.add(resblock.forward(x.clone(), None));
                }
            }
            x = xs / self.num_kernels as f32;
        }
        x = leaky_relu(x, 0.01);
        x = self.conv_post.forward(x);
        x = x.tanh();
        x
    }
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    resblock_type: usize,
    channels: usize,
    kernel_size: usize,
    dilation: Vec<usize>,
}
impl ResBlockConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> ResBlock<B> {
        if self.resblock_type == 1 {
            ResBlock::ResBlock1(
                ResBlock1Config::new(self.channels, self.kernel_size, self.dilation.clone())
                    .init(device),
            )
        } else {
            ResBlock::ResBlock2(
                ResBlock2Config::new(self.channels, self.kernel_size, self.dilation.clone())
                    .init(device),
            )
        }
    }
}

#[derive(Module, Debug)]
enum ResBlock<B: Backend> {
    ResBlock1(ResBlock1<B>),
    ResBlock2(ResBlock2<B>),
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, x_mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        match self {
            ResBlock::ResBlock1(res_block1) => res_block1.forward(x, x_mask),
            ResBlock::ResBlock2(res_block2) => res_block2.forward(x, x_mask),
        }
    }
}

#[derive(Config, Debug)]
pub struct ResBlock1Config {
    channels: usize,
    kernel_size: usize,
    dilation: Vec<usize>,
}
impl ResBlock1Config {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> ResBlock1<B> {
        ResBlock1 {
            convs1: vec![
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(self.dilation[0])
                    .with_padding(get_padding(self.kernel_size, self.dilation[0]))
                    .init(device),
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(self.dilation[1])
                    .with_padding(get_padding(self.kernel_size, self.dilation[1]))
                    .init(device),
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(self.dilation[2])
                    .with_padding(get_padding(self.kernel_size, self.dilation[2]))
                    .init(device),
            ],
            //self.convs1.apply(init_weights)
            convs2: vec![
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(1)
                    .with_padding(get_padding(self.kernel_size, 1))
                    .init(device),
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(1)
                    .with_padding(get_padding(self.kernel_size, 1))
                    .init(device),
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(1)
                    .with_padding(get_padding(self.kernel_size, 1))
                    .init(device),
            ],
            //self.convs2.apply(init_weights)
            lrelu_slope: 0.1,
        }
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

#[derive(Module, Debug)]
pub struct ResBlock1<B: Backend> {
    convs1: Vec<WNConv1d<B>>,
    convs2: Vec<WNConv1d<B>>,
    lrelu_slope: f64,
}

impl<B: Backend> ResBlock1<B> {
    pub fn forward(&self, mut x: Tensor<B, 3>, x_mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = leaky_relu(x.clone(), self.lrelu_slope);
            let xt = match x_mask {
                Some(ref x_mask) => xt * x_mask.clone(),
                None => xt,
            };
            let xt = c1.forward(xt);
            let xt = leaky_relu(xt, self.lrelu_slope);
            let xt = match x_mask {
                Some(ref x_mask) => xt * x_mask.clone(),
                None => xt,
            };
            let xt = c2.forward(xt);
            x = xt + x;
        }

        x = match x_mask {
            Some(ref x_mask) => x * x_mask.clone(),
            None => x,
        };
        x
    }
}

#[derive(Config, Debug)]
pub struct ResBlock2Config {
    channels: usize,
    kernel_size: usize,
    dilation: Vec<usize>,
}
impl ResBlock2Config {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> ResBlock2<B> {
        ResBlock2 {
            convs: vec![
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(self.dilation[0])
                    .with_padding(get_padding(self.kernel_size, self.dilation[0]))
                    .init(device),
                WNConv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_stride(1)
                    .with_dilation(self.dilation[1])
                    .with_padding(get_padding(self.kernel_size, self.dilation[1]))
                    .init(device),
            ],
            lrelu_slope: 0.1,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResBlock2<B: Backend> {
    convs: Vec<WNConv1d<B>>,
    lrelu_slope: f64,
}

impl<B: Backend> ResBlock2<B> {
    #[trace_shapes]
    pub fn forward(&self, mut x: Tensor<B, 3>, x_mask: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        for c in self.convs.iter() {
            let xt = leaky_relu(x.clone(), self.lrelu_slope);
            let xt = match x_mask {
                Some(ref x_mask) => xt * x_mask.clone(),
                None => xt,
            };
            let xt = c.forward(xt);
            x = x + xt;
        }
        x = match x_mask {
            Some(x_mask) => x * x_mask,
            None => x,
        };
        x
    }
}

#[derive(Config, Debug)]
pub struct PosteriorEncoderConfig {
    in_channels: usize,
    out_channels: usize,
    hidden_channels: usize,
    kernel_size: usize,
    dilation_rate: usize,
    n_layers: usize,
    gin_channels: usize,
}

impl PosteriorEncoderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PosteriorEncoder<B> {
        PosteriorEncoder {
            pre: Conv1dConfig::new(self.in_channels, self.hidden_channels, 1).init(device),
            enc: WNConfig::new(
                self.hidden_channels,
                self.kernel_size,
                self.dilation_rate,
                self.n_layers,
            )
            .with_gin_channels(self.gin_channels)
            .init(device),
            proj: Conv1dConfig::new(self.hidden_channels, self.out_channels * 2, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct PosteriorEncoder<B: Backend> {
    pre: Conv1d<B>,
    enc: WN<B>,
    proj: Conv1d<B>,
}

#[derive(Config, Debug)]
pub struct ResidualCouplingBlockConfig {
    channels: usize,
    hidden_channels: usize,
    kernel_size: usize,
    dilation_rate: usize,
    n_layers: usize,
    #[config(default = 4)]
    n_flows: usize,
    gin_channels: usize,
}
impl ResidualCouplingBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResidualCouplingBlock<B> {
        let mut flows = vec![];
        for i in 0..self.n_flows {
            flows.push(ResidualCouplingBlockLayers::ResidualCouplingLayer(
                ResidualCouplingLayerConfig::new(
                    self.channels,
                    self.hidden_channels,
                    self.kernel_size,
                    self.dilation_rate,
                    self.n_layers,
                )
                .with_gin_channels(self.gin_channels)
                .with_mean_only(true)
                .init(device),
            ));
            flows.push(ResidualCouplingBlockLayers::Flip(Flip {}));
        }
        ResidualCouplingBlock { flows }
    }
}

#[derive(Module, Debug)]
pub struct ResidualCouplingBlock<B: Backend> {
    flows: Vec<ResidualCouplingBlockLayers<B>>,
}

impl<B: Backend> ResidualCouplingBlock<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
    ) -> Tensor<B, 3> {
        if !reverse {
            for flow in self.flows.iter() {
                (x, _) = flow.forward(x, x_mask.clone(), g.clone(), reverse);
            }
        } else {
            let mut flows = self.flows.clone();
            flows.reverse();
            for flow in flows {
                (x, _) = flow.forward(x, x_mask.clone(), g.clone(), reverse)
            }
        }

        x
    }
}

#[derive(Module, Debug)]
enum ResidualCouplingBlockLayers<B: Backend> {
    ResidualCouplingLayer(ResidualCouplingLayer<B>),
    Flip(Flip),
}

impl<B: Backend> ResidualCouplingBlockLayers<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
    ) -> (Tensor<B, 3>, f32) {
        match self {
            ResidualCouplingBlockLayers::ResidualCouplingLayer(residual_coupling_layer) => {
                residual_coupling_layer.forward(x, x_mask, g, reverse)
            }
            ResidualCouplingBlockLayers::Flip(flip) => flip.forward(x, reverse),
        }
    }
}

#[derive(Config, Debug)]
pub struct ResidualCouplingLayerConfig {
    channels: usize,
    hidden_channels: usize,
    kernel_size: usize,
    dilation_rate: usize,
    n_layers: usize,
    #[config(default = "0.0")]
    p_dropout: f64,
    #[config(default = 0)]
    gin_channels: usize,
    #[config(default = false)]
    mean_only: bool,
}
impl ResidualCouplingLayerConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> ResidualCouplingLayer<B> {
        assert!(self.channels % 2 == 0, "channels should be divisible by 2");
        let half_channels = self.channels / 2;

        ResidualCouplingLayer {
            pre: Conv1dConfig::new(half_channels, self.hidden_channels, 1).init(device),
            enc: WNConfig::new(
                self.hidden_channels,
                self.kernel_size,
                self.dilation_rate,
                self.n_layers,
            )
            .with_p_dropout(self.p_dropout)
            .with_gin_channels(self.gin_channels)
            .init(device),
            post: Conv1dConfig::new(
                self.hidden_channels,
                half_channels * (2 - if self.mean_only { 1 } else { 0 }),
                1,
            )
            .init(device),
            half_channels,
            mean_only: self.mean_only,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualCouplingLayer<B: Backend> {
    pre: Conv1d<B>,
    enc: WN<B>,
    post: Conv1d<B>,
    half_channels: usize,
    mean_only: bool,
}

impl<B: Backend> ResidualCouplingLayer<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
    ) -> (Tensor<B, 3>, f32) {
        let ret = x.split_with_sizes(vec![self.half_channels, self.half_channels], 1);
        let x0 = ret[0].clone();
        let x1 = ret[1].clone();
        let h = self.pre.forward(x0.clone()) * x_mask.clone();
        let h = self.enc.forward(h, x_mask.clone(), g);
        let stats = self.post.forward(h) * x_mask.clone();

        let (m, logs) = if !self.mean_only {
            let ret = stats.split_with_sizes(vec![self.half_channels, self.half_channels], 1);
            (ret[0].clone(), ret[1].clone())
        } else {
            (stats.clone(), stats.zeros_like())
        };

        if !reverse {
            let x1 = m + x1 * logs.clone().exp() * x_mask;
            let x = Tensor::cat(vec![x0, x1], 1);
            let logdet = logs.sum_dims(&[1, 2]);
            (x, logdet.squeeze::<1>().to_data().to_vec().unwrap()[0])
        } else {
            let x1 = (x1 - m) * (-logs).exp() * x_mask;
            let x = Tensor::cat(vec![x0, x1], 1);
            (x, 0.0)
        }
    }
}

#[derive(Config, Debug)]
pub struct StochasticDurationPredictorConfig {
    in_channels: usize,
    filter_channels: usize,
    kernel_size: usize,
    p_dropout: f64,
    n_flows: usize,
    gin_channels: usize,
}
impl StochasticDurationPredictorConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> StochasticDurationPredictor<B> {
        let mut flows = vec![StochasticDurationPredictorLayers::ElementwiseAffine(
            ElementwiseAffineConfig::new(2).init(device),
        )];

        for i in 0..self.n_flows {
            flows.push(StochasticDurationPredictorLayers::ConvFlow(
                ConvFlowConfig::new(2, self.filter_channels, self.kernel_size, 3).init(device),
            ));
            flows.push(StochasticDurationPredictorLayers::Flip(Flip {}))
        }

        let mut post_flows = vec![StochasticDurationPredictorLayers::ElementwiseAffine(
            ElementwiseAffineConfig::new(2).init(device),
        )];

        for i in 0..4 {
            post_flows.push(StochasticDurationPredictorLayers::ConvFlow(
                ConvFlowConfig::new(2, self.filter_channels, self.kernel_size, 3).init(device),
            ));
            post_flows.push(StochasticDurationPredictorLayers::Flip(Flip {}))
        }

        StochasticDurationPredictor {
            log_flow: Log {},
            flows,
            post_pre: Conv1dConfig::new(1, self.filter_channels, 1).init(device),
            post_proj: Conv1dConfig::new(self.filter_channels, self.filter_channels, 1)
                .init(device),
            post_convs: DDSConvConfig::new(self.filter_channels, self.kernel_size, 3)
                .with_p_dropout(self.p_dropout)
                .init(device),
            post_flows,
            pre: Conv1dConfig::new(self.in_channels, self.filter_channels, 1).init(device),
            proj: Conv1dConfig::new(self.filter_channels, self.filter_channels, 1).init(device),
            convs: DDSConvConfig::new(self.filter_channels, self.kernel_size, 3)
                .with_p_dropout(self.p_dropout)
                .init(device),
            cond: if self.gin_channels != 0 {
                Some(Conv1dConfig::new(self.gin_channels, self.filter_channels, 1).init(device))
            } else {
                None
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct StochasticDurationPredictor<B: Backend> {
    pub log_flow: Log,
    pub flows: Vec<StochasticDurationPredictorLayers<B>>,
    pub post_pre: Conv1d<B>,
    pub post_proj: Conv1d<B>,
    pub post_convs: DDSConv<B>,
    pub post_flows: Vec<StochasticDurationPredictorLayers<B>>,
    pub pre: Conv1d<B>,
    pub proj: Conv1d<B>,
    pub convs: DDSConv<B>,
    pub cond: Option<Conv1d<B>>,
}

impl<B: Backend> StochasticDurationPredictor<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        w: Option<Tensor<B, 3>>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
        noise_scale: f64,
    ) -> Tensor<B, 3> {
        let x = self.pre.forward(x);
        let x = if let Some(g) = g.as_ref() {
            x + g.clone()
        } else {
            x
        };
        let x = self.convs.forward(x, x_mask.clone(), g.clone());
        let x = self.proj.forward(x) * x_mask.clone();

        if !reverse {
            let mut logdet_tot_q: f32 = 0.0;
            let w = w.expect("w must not be none");
            let h_w = self.post_pre.forward(w.clone());
            let h_w = self.post_convs.forward(h_w, x_mask.clone(), g.clone());
            let h_w = self.post_proj.forward(h_w) * x_mask.clone();
            let e_q: Tensor<B, 3> = Tensor::random(
                [w.dims()[0], 2, w.dims()[2]],
                Default::default(),
                &x_mask.device(),
            ) * x_mask.clone();
            let mut z_q = e_q.clone();

            for flow in self.post_flows.iter() {
                let (t_z_q, logdet_q) =
                    flow.forward(z_q, x_mask.clone(), Some(x.clone() + h_w.clone()), false);
                z_q = t_z_q;
                logdet_tot_q += logdet_q;
            }
            let ret = z_q.split_with_sizes(vec![1, 1], 1);
            let z_u = ret[0].clone();
            let z1 = ret[1].clone();
            let u = sigmoid(z_u.clone()) * x_mask.clone();
            let z0 = (w - u) * x_mask.clone();
            logdet_tot_q += (log_sigmoid(z_u.clone()) + log_sigmoid(z_u.neg()) * x_mask.clone())
                .sum_dims(&[1, 2])
                .to_data()
                .to_vec::<f32>()
                .unwrap()[0];
            let logq: Tensor<B, _> =
                (-0.5_f64 * ((2.0 * PI).ln() + e_q.powi_scalar(2)) * x_mask.clone())
                    .sum_dims(&[1, 2])
                    - logdet_tot_q;
            let mut logdet_tot: f32 = 0.0;
            let (z0, logdet) = self.log_flow.forward(z0, x_mask.clone(), false);
            logdet_tot += logdet;
            let mut z = Tensor::cat(vec![z0, z1], 1);
            for flow in self.flows.iter() {
                let (t_z, logdet) = flow.forward(z, x_mask.clone(), g.clone(), reverse);
                z = t_z;
                logdet_tot += logdet;
            }
            let nll: Tensor<B, _> = (-0.5_f64 * ((2.0 * PI).ln() + z.powi_scalar(2)) * x_mask)
                .sum_dims(&[1, 2])
                - logdet_tot;
            return nll + logq;
        } else {
            let mut flows = self.flows.clone();
            flows.reverse();
            let flows = [&flows[..flows.len() - 2], &flows[(flows.len() - 1)..]].concat();
            let mut z = Tensor::random(
                [x.dims()[0], 2, x.dims()[2]],
                Default::default(),
                &x.device(),
            ) * noise_scale;

            for flow in flows {
                let (t_z, _) = flow.forward(z, x_mask.clone(), Some(x.clone()), reverse);
                z = t_z;
            }
            let ret = z.split_with_sizes(vec![1, 1], 1);
            let z0 = ret[0].clone();
            return z0;
        }
    }
}

#[derive(Module, Debug)]
pub enum StochasticDurationPredictorLayers<B: Backend> {
    ElementwiseAffine(ElementwiseAffine<B>),
    ConvFlow(ConvFlow<B>),
    Flip(Flip),
}

impl<B: Backend> StochasticDurationPredictorLayers<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
    ) -> (Tensor<B, 3>, f32) {
        match self {
            StochasticDurationPredictorLayers::ElementwiseAffine(elementwise_affine) => {
                elementwise_affine.forward(x, x_mask, reverse)
            }
            StochasticDurationPredictorLayers::ConvFlow(conv_flow) => {
                conv_flow.forward(x, x_mask, g, reverse)
            }
            StochasticDurationPredictorLayers::Flip(flip) => flip.forward(x, reverse),
        }
    }
}

#[derive(Config, Debug)]
pub struct ElementwiseAffineConfig {
    channels: usize,
}

impl ElementwiseAffineConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> ElementwiseAffine<B> {
        ElementwiseAffine {
            m: Param::from_tensor(Tensor::zeros([self.channels, 1], device)),
            logs: Param::from_tensor(Tensor::zeros([self.channels, 1], device)),
        }
    }
}

#[derive(Module, Debug)]
pub struct ElementwiseAffine<B: Backend> {
    m: Param<Tensor<B, 2>>,
    logs: Param<Tensor<B, 2>>,
}

impl<B: Backend> ElementwiseAffine<B> {
    #[trace_shapes]
    fn forward(&self, x: Tensor<B, 3>, x_mask: Tensor<B, 3>, reverse: bool) -> (Tensor<B, 3>, f32) {
        if !reverse {
            let y =
                self.m.val().clone().unsqueeze() + self.logs.val().clone().exp().unsqueeze() * x;
            let y = y * x_mask.clone();
            let logdet = (self.logs.val().clone().unsqueeze() * x_mask.clone())
                .sum_dims(&[1, 2])
                .squeeze::<1>()
                .to_data()
                .to_vec()
                .unwrap()[0];
            return (y, logdet);
        } else {
            let x =
                (x - self.m.val().unsqueeze()) * (self.logs.val().neg().unsqueeze()).exp() * x_mask;
            return (x, 0.0);
        }
    }
}

#[derive(Config, Debug)]
pub struct ConvFlowConfig {
    in_channels: usize,
    filter_channels: usize,
    kernel_size: usize,
    n_layers: usize,
    #[config(default = 10)]
    num_bins: usize,
    #[config(default = "5.0")]
    tail_bound: f64,
}
impl ConvFlowConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> ConvFlow<B> {
        let half_channels = self.in_channels / 2;
        ConvFlow {
            pre: Conv1dConfig::new(half_channels, self.filter_channels, 1).init(device),
            convs: DDSConvConfig::new(self.filter_channels, self.kernel_size, self.n_layers)
                .with_p_dropout(0.0)
                .init(device),
            proj: Conv1dConfig::new(
                self.filter_channels,
                half_channels * (self.num_bins * 3 - 1),
                1,
            )
            .init(device),
            half_channels,
            num_bins: self.num_bins,
            filter_channels: self.filter_channels,
            tail_bound: self.tail_bound,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvFlow<B: Backend> {
    pre: Conv1d<B>,
    convs: DDSConv<B>,
    proj: Conv1d<B>,
    half_channels: usize,
    num_bins: usize,
    filter_channels: usize,
    tail_bound: f64,
}

impl<B: Backend> ConvFlow<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
        reverse: bool,
    ) -> (Tensor<B, 3>, f32) {
        let ret = x.split_with_sizes(vec![self.half_channels, self.half_channels], 1);
        let x0 = ret[0].clone();
        let x1 = ret[1].clone();
        let h = self.pre.forward(x0.clone());
        let h = self.convs.forward(h, x_mask.clone(), g);
        let h = self.proj.forward(h) * x_mask.clone();

        let [b, c, t] = x0.dims();
        let h = h
            .reshape([b as isize, c as isize, -1, t as isize])
            .permute([0, 1, 3, 2]);

        let unnormalized_widths = h
            .clone()
            .slice([s![..], s![..], s![..], s![..self.num_bins]])
            / (self.filter_channels as f64).sqrt();
        let unnormalized_heights =
            h.clone()
                .slice([s![..], s![..], s![..], s![self.num_bins..2 * self.num_bins]])
                / (self.filter_channels as f64).sqrt();
        let unnormalized_derivatives = h.slice([s![..], s![..], s![..], s![2 * self.num_bins..]]);

        let (x1, logabsdet) = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse,
            Some("linear"),
            self.tail_bound,
        );

        let x = Tensor::cat(vec![x0, x1], 1) * x_mask.clone();

        let logdet = (logabsdet * x_mask)
            .sum_dims(&[1, 2])
            .to_data()
            .to_vec()
            .unwrap()[0];

        if !reverse { (x, logdet) } else { (x, 0.0) }
    }
}

#[derive(Config, Debug)]
pub struct DurationPredictorConfig {
    in_channels: usize,
    filter_channels: usize,
    kernel_size: usize,
    p_dropout: f32,
    gin_channels: usize,
}
impl DurationPredictorConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DurationPredictor<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct DurationPredictor<B: Backend> {
    drop: Dropout,
    conv_1: Conv1d<B>,
    norm_1: LayerNorm<B>,
    conv_2: Conv1d<B>,
    norm_2: LayerNorm<B>,
    proj: Conv1d<B>,
    cond: Option<Conv1d<B>>,
}
impl<B: Backend> DurationPredictor<B> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Config, Debug)]
pub struct DDSConvConfig {
    channels: usize,
    kernel_size: usize,
    n_layers: usize,
    #[config(default = "0.0")]
    p_dropout: f64,
}
impl DDSConvConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> DDSConv<B> {
        let mut convs_sep = vec![];
        let mut convs_1x1 = vec![];
        let mut norms_1 = vec![];
        let mut norms_2 = vec![];
        for i in 0..self.n_layers {
            let dilation = self.kernel_size.pow(i as u32);
            let padding = (self.kernel_size * dilation - dilation) / 2;
            convs_sep.push(
                Conv1dConfig::new(self.channels, self.channels, self.kernel_size)
                    .with_groups(self.channels)
                    .with_dilation(dilation)
                    .with_padding(PaddingConfig1d::Explicit(padding, padding))
                    .init(device),
            );
            convs_1x1.push(Conv1dConfig::new(self.channels, self.channels, 1).init(device));
            norms_1.push(LayerNormConfig::new(self.channels).init(device));
            norms_2.push(LayerNormConfig::new(self.channels).init(device));
        }

        DDSConv {
            drop: DropoutConfig::new(self.p_dropout).init(),
            convs_sep,
            convs_1x1,
            norms_1,
            norms_2,
            n_layers: self.n_layers,
        }
    }
}

#[derive(Module, Debug)]
pub struct DDSConv<B: Backend> {
    drop: Dropout,
    convs_sep: Vec<Conv1d<B>>,
    convs_1x1: Vec<Conv1d<B>>,
    norms_1: Vec<LayerNorm<B>>,
    norms_2: Vec<LayerNorm<B>>,
    n_layers: usize,
}

impl<B: Backend> DDSConv<B> {
    //'input': [(1, 192, 123), (1, 1, 123)], 'output': (1, 192, 123)
    #[trace_shapes]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let mut x = match g {
            Some(g) => x + g,
            None => x,
        };
        for i in 0..self.n_layers {
            let y = self.convs_sep[i].forward(x.clone() * x_mask.clone());
            //TODO check if swap_dims is correct here
            let y = self.norms_1[i].forward(y.swap_dims(1, 2)).swap_dims(2, 1);
            let y = gelu(y);
            let y = self.convs_1x1[i].forward(y);
            let y = self.norms_2[i].forward(y.swap_dims(1, 2)).swap_dims(2, 1);
            let y = gelu(y);
            let y = self.drop.forward(y);
            x = x + y
        }
        x * x_mask
    }
}

#[derive(Config, Debug)]
pub struct WNConfig {
    hidden_channels: usize,
    kernel_size: usize,
    dilation_rate: usize,
    n_layers: usize,
    #[config(default = 0)]
    gin_channels: usize,
    #[config(default = "0.0")]
    p_dropout: f64,
}
impl WNConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> WN<B> {
        assert!(self.kernel_size % 2 == 1);
        let cond_layer = if self.gin_channels != 0 {
            Some(
                WNConv1dConfig::new(
                    self.gin_channels,
                    2 * self.hidden_channels * self.n_layers,
                    1,
                )
                .init(device),
            )
        } else {
            None
        };

        let mut in_layers = vec![];
        let mut res_skip_layers = vec![];

        for i in 0..self.n_layers {
            let dilation = self.dilation_rate.pow(i as u32);
            let padding = (self.kernel_size * dilation - dilation) / 2;
            in_layers.push(
                WNConv1dConfig::new(
                    self.hidden_channels,
                    2 * self.hidden_channels,
                    self.kernel_size,
                )
                .with_dilation(dilation)
                .with_padding(padding)
                .init(device),
            );

            let res_skip_channels = if i < self.n_layers - 1 {
                2 * self.hidden_channels
            } else {
                self.hidden_channels
            };

            res_skip_layers
                .push(WNConv1dConfig::new(self.hidden_channels, res_skip_channels, 1).init(device));
        }

        WN {
            in_layers,
            res_skip_layers,
            drop: DropoutConfig::new(self.p_dropout).init(),
            cond_layer,
            hidden_channels: self.hidden_channels,
            n_layers: self.n_layers,
        }
    }
}

#[derive(Module, Debug)]
pub struct WN<B: Backend> {
    in_layers: Vec<WNConv1d<B>>,
    res_skip_layers: Vec<WNConv1d<B>>,
    drop: Dropout,
    cond_layer: Option<WNConv1d<B>>,
    hidden_channels: usize,
    n_layers: usize,
}

impl<B: Backend> WN<B> {
    #[trace_shapes]
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        g: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let mut output = x.zeros_like();

        let g = match (g, self.cond_layer.as_ref()) {
            (Some(g), Some(cond_layer)) => Some(cond_layer.forward(g)),
            _ => None,
        };

        for i in 0..self.n_layers {
            let x_in = self.in_layers[i].forward(x.clone());
            let g_l = match g {
                Some(ref g) => {
                    let cond_offset = i * 2 * self.hidden_channels;
                    g.clone().slice([
                        s![..],
                        s![cond_offset..cond_offset + 2 * self.hidden_channels],
                        s![..],
                    ])
                }
                None => x_in.zeros_like(),
            };
            let acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels);
            let acts = self.drop.forward(acts);

            let res_skip_acts = self.res_skip_layers[i].forward(acts.clone());

            output = if i < self.n_layers - 1 {
                let res_acts =
                    res_skip_acts
                        .clone()
                        .slice([s![..], s![..self.hidden_channels], s![..]]);
                x = (x + res_acts.clone()) * x_mask.clone();
                output + res_skip_acts.slice([s![..], s![self.hidden_channels..], s![..]])
            } else {
                output + res_skip_acts
            };
        }

        output * x_mask
    }
}

fn fused_add_tanh_sigmoid_multiply<B: Backend>(
    input_a: Tensor<B, 3>,
    input_b: Tensor<B, 3>,
    n_channels_int: usize,
) -> Tensor<B, 3> {
    let in_act = input_a + input_b;
    let t_act = in_act
        .clone()
        .slice([s![..], s![..n_channels_int], s![..]])
        .tanh();
    let s_act = sigmoid(in_act.clone().slice([s![..], s![n_channels_int..], s![..]]));
    t_act * s_act
}

#[derive(Config, Debug)]
pub struct MultiPeriodDiscriminatorConfig {
    #[config(default = false)]
    use_spectral_norm: bool,
}

impl MultiPeriodDiscriminatorConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiPeriodDiscriminator<B> {
        let periods = [2, 3, 5, 7, 11];

        let mut discriminators = vec![MultiPeriodDiscriminatorLayers::DiscriminatorS(
            DiscriminatorSConfig::new()
                .with_use_spectral_norm(self.use_spectral_norm)
                .init(device),
        )];

        for i in periods {
            discriminators.push(MultiPeriodDiscriminatorLayers::DiscriminatorP(
                DiscriminatorPConfig::new(i)
                    .with_use_spectral_norm(self.use_spectral_norm)
                    .init(device),
            ));
        }
        MultiPeriodDiscriminator { discriminators }
    }
}

#[derive(Module, Debug)]
pub struct MultiPeriodDiscriminator<B: Backend> {
    discriminators: Vec<MultiPeriodDiscriminatorLayers<B>>,
}

#[derive(Module, Debug)]
enum MultiPeriodDiscriminatorLayers<B: Backend> {
    DiscriminatorS(DiscriminatorS<B>),
    DiscriminatorP(DiscriminatorP<B>),
}

#[derive(Config, Debug)]
pub struct DiscriminatorSConfig {
    #[config(default = false)]
    use_spectral_norm: bool,
}
impl DiscriminatorSConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> DiscriminatorS<B> {
        // might need to handle use_spectral_norm later, but it looks like piper is not using it
        DiscriminatorS {
            convs: vec![
                DiscriminatorSLayers::WNConv1d(
                    WNConv1dConfig::new(1, 16, 15)
                        .with_stride(1)
                        .with_padding(7)
                        .init(device),
                ),
                DiscriminatorSLayers::WNConv1d(
                    WNConv1dConfig::new(16, 64, 41)
                        .with_stride(4)
                        .with_padding(20)
                        .with_groups(4)
                        .init(device),
                ),
                DiscriminatorSLayers::WNConv1d(
                    WNConv1dConfig::new(64, 256, 41)
                        .with_stride(4)
                        .with_padding(20)
                        .with_groups(16)
                        .init(device),
                ),
                DiscriminatorSLayers::WNConv1d(
                    WNConv1dConfig::new(256, 1024, 41)
                        .with_stride(4)
                        .with_padding(20)
                        .with_groups(64)
                        .init(device),
                ),
                DiscriminatorSLayers::WNConv1d(
                    WNConv1dConfig::new(1024, 1024, 41)
                        .with_stride(4)
                        .with_padding(20)
                        .with_groups(256)
                        .init(device),
                ),
                DiscriminatorSLayers::WNConv1d(
                    WNConv1dConfig::new(1024, 1024, 5)
                        .with_stride(1)
                        .with_padding(2)
                        .init(device),
                ),
            ],
            conv_post: DiscriminatorSLayers::WNConv1d(
                WNConv1dConfig::new(1024, 1, 3)
                    .with_stride(1)
                    .with_padding(1)
                    .init(device),
            ),
        }
    }
}

#[derive(Module, Debug)]
pub struct DiscriminatorS<B: Backend> {
    convs: Vec<DiscriminatorSLayers<B>>,
    conv_post: DiscriminatorSLayers<B>,
}

#[derive(Module, Debug)]
enum DiscriminatorSLayers<B: Backend> {
    SNConv1d(SNConv1d<B>),
    WNConv1d(WNConv1d<B>),
}

#[derive(Config, Debug)]
pub struct DiscriminatorPConfig {
    period: usize,
    #[config(default = 5)]
    kernel_size: usize,
    #[config(default = 3)]
    stride: usize,
    #[config(default = false)]
    use_spectral_norm: bool,
}
impl DiscriminatorPConfig {
    fn init<B: Backend>(&self, device: &<B as Backend>::Device) -> DiscriminatorP<B> {
        let padding = get_padding(self.kernel_size, 1);
        DiscriminatorP {
            convs: vec![
                DiscriminatorPLayers::WNConv2d(
                    WNConv2dConfig::new([1, 32], [self.kernel_size, 1])
                        .with_stride([self.stride, 1])
                        .with_padding(PaddingConfig2d::Explicit(padding, 0, padding, 0))
                        .init(device),
                ),
                DiscriminatorPLayers::WNConv2d(
                    WNConv2dConfig::new([32, 128], [self.kernel_size, 1])
                        .with_stride([self.stride, 1])
                        .with_padding(PaddingConfig2d::Explicit(padding, 0, padding, 0))
                        .init(device),
                ),
                DiscriminatorPLayers::WNConv2d(
                    WNConv2dConfig::new([128, 512], [self.kernel_size, 1])
                        .with_stride([self.stride, 1])
                        .with_padding(PaddingConfig2d::Explicit(padding, 0, padding, 0))
                        .init(device),
                ),
                DiscriminatorPLayers::WNConv2d(
                    WNConv2dConfig::new([512, 1024], [self.kernel_size, 1])
                        .with_stride([self.stride, 1])
                        .with_padding(PaddingConfig2d::Explicit(padding, 0, padding, 0))
                        .init(device),
                ),
                DiscriminatorPLayers::WNConv2d(
                    WNConv2dConfig::new([1024, 1024], [self.kernel_size, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Explicit(padding, 0, padding, 0))
                        .init(device),
                ),
            ],
            conv_post: WNConv2dConfig::new([1024, 1], [3, 1])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DiscriminatorP<B: Backend> {
    convs: Vec<DiscriminatorPLayers<B>>,
    conv_post: WNConv2d<B>,
}

#[derive(Module, Debug)]
enum DiscriminatorPLayers<B: Backend> {
    SNConv2d(SNConv2d<B>),
    WNConv2d(WNConv2d<B>),
}

#[derive(Debug, Config)]
pub struct WNConvTranspose1dConfig {
    input_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    stride: usize,
    #[config(default = 0)]
    padding: usize,
    #[config(default = 0)]
    output_padding: usize,
    #[config(default = true)]
    bias: bool,
}

impl WNConvTranspose1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNConvTranspose1d<B> {
        let conv = ConvTranspose1dConfig::new([self.input_dim, self.output_dim], self.kernel_size)
            .with_stride(self.stride)
            .with_padding(self.padding)
            .with_padding_out(self.output_padding)
            .init(device);

        let v = conv.weight.clone();

        // Initialize g as the L2 norm of each input filter
        // weight shape for transposed conv: [channels_in, channels_out/groups, kernel_size]
        let g = Param::from_tensor(
            v.val()
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .sum_dim(1)
                .sqrt(),
        );

        WNConvTranspose1d {
            weight_v: v,
            weight_g: g,
            bias: if self.bias { conv.bias } else { None },
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNConvTranspose1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
}

impl<B: Backend> WNConvTranspose1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let v = self.weight_v.val();

        // Compute L2 norm of v per input channel
        // v shape: [channels_in, channels_out/groups, kernel_size]
        // norm shape: [channels_in, 1, 1]
        let norm = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();

        // Normalize: v_hat = v / ||v||
        let v_hat = v / (norm + 1e-12);

        // Weight normalization: w = g * v_hat
        let w = self.weight_g.val() * v_hat;

        conv_transpose1d(
            x,
            w,
            self.bias.clone().map(|b| b.val()),
            ConvTransposeOptions::<1>::new(
                [self.stride],
                [self.padding],
                [self.output_padding],
                [1], // dilation
                1,   // groups
            ),
        )
    }
}

#[derive(Debug, Config)]
pub struct WNConv1dConfig {
    channels_in: usize,
    channels_out: usize,
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    dilation: usize,
    #[config(default = 1)]
    groups: usize,
    #[config(default = 0)]
    padding: usize,
    #[config(default = true)]
    bias: bool,
}

impl WNConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNConv1d<B> {
        use burn::nn::conv::Conv1dConfig;

        let conv = Conv1dConfig::new(self.channels_in, self.channels_out, self.kernel_size)
            .with_groups(self.groups)
            .init(device);

        let v = conv.weight.clone();

        // Initialize g as the norm of each output filter: g[i] = ||v[i]||
        // v shape: [channels_out, channels_in/groups, kernel_size]
        let g = Param::from_tensor(
            v.val()
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .sum_dim(1)
                .sqrt(),
        );

        WNConv1d {
            weight_v: v,
            weight_g: g,
            bias: if self.bias { conv.bias } else { None },
            stride: self.stride,
            dilation: self.dilation,
            groups: self.groups,
            padding: self.padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct WNConv1d<B: Backend> {
    pub weight_g: Param<Tensor<B, 3>>,
    pub weight_v: Param<Tensor<B, 3>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl<B: Backend> WNConv1d<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let v = self.weight_v.val();

        // Compute the L2 norm of v per output channel: ||v[i]||
        // v shape: [channels_out, channels_in/groups, kernel_size]
        // norm shape: [channels_out, 1, 1]
        let norm = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();

        // Normalize: v_hat = v / ||v||
        let v_hat = v / (norm.clone() + 1e-12);

        // Weight normalization: w = g * v_hat
        let w = self.weight_g.val() * v_hat;

        // Apply causal (left-only) padding, then conv with padding=0
        let x = x.pad((self.padding, self.padding, 0, 0), PadMode::Constant(0.0));

        conv1d(
            x,
            w,
            self.bias.clone().map(|b| b.val()),
            ConvOptions::<1>::new([self.stride], [0], [self.dilation], self.groups),
        )
    }
}

#[derive(Module, Debug)]
pub struct SNConv1d<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Option<Param<Tensor<B, 1>>>,
}

#[derive(Config, Debug)]
pub struct WNConv2dConfig {
    /// `[channels_in, channels_out]`
    pub channels: [usize; 2],
    /// `[kernel_h, kernel_w]`
    pub kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    pub stride: [usize; 2],
    #[config(default = "[1, 1]")]
    pub dilation: [usize; 2],
    #[config(default = "1")]
    pub groups: usize,
    #[config(default = "PaddingConfig2d::Valid")]
    pub padding: PaddingConfig2d,
    #[config(default = true)]
    pub bias: bool,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0),fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl WNConv2dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WNConv2d<B> {
        let [ch_in, ch_out] = self.channels;
        let [kh, kw] = self.kernel_size;

        let weight_shape = [ch_out, ch_in / self.groups, kh, kw];
        let fan_in = ch_in / self.groups * kh * kw;
        let fan_out = ch_out / self.groups * kh * kw;

        // v — direction tensor, same shape as a plain Conv2d weight
        let weight_v: Param<Tensor<B, 4>> =
            self.initializer
                .init_with(weight_shape, Some(fan_in), Some(fan_out), device);

        // g — per-output-channel magnitude, initialised to ‖v_i‖ so the
        //     very first forward pass is identical to an unnormalised conv.
        //
        // Flatten each filter to a vector [out_ch, in/g * kH * kW],
        // then take its L2 norm → [out_ch], then reshape → [out_ch, 1, 1, 1].
        let num_elements = (ch_in / self.groups) * kh * kw; // elements per filter
        let g_init = weight_v
            .val() // [out_ch, in/g, kH, kW]
            .reshape([ch_out, num_elements]) // [out_ch, in/g*kH*kW]
            .powf_scalar(2.0_f32) // element-wise square
            .sum_dim(1) // [out_ch, 1]  (keepdim)
            .sqrt() // [out_ch, 1]
            .reshape([ch_out, 1, 1, 1]); // [out_ch, 1, 1, 1]  ✓

        let weight_g: Param<Tensor<B, 4>> = Param::from_tensor(g_init);

        let bias = if self.bias {
            Some(
                self.initializer
                    .init_with([ch_out], Some(fan_in), Some(fan_out), device),
            )
        } else {
            None
        };

        WNConv2d {
            weight_v,
            weight_g,
            bias,
            stride: self.stride,
            kernel_size: self.kernel_size,
            dilation: self.dilation,
            padding: Ignored(self.padding.clone()),
            groups: self.groups,
        }
    }
}

/// 2-D convolution with weight normalisation.
///
/// Equivalent to PyTorch's:
/// ```python
/// weight_norm(
///     Conv2d(1, 32, (kernel_size, 1), (stride, 1),
///            padding=(get_padding(kernel_size, 1), 0))
/// )
/// ```
///
/// Weight normalisation reparametrises the weight tensor `w` as:
///
/// ```text
/// w = g * (v / ‖v‖)
/// ```
///
/// where `g` (shape `[out_ch, 1, 1, 1]`) is a learnable magnitude and
/// `v` (shape `[out_ch, in_ch/groups, kH, kW]`) is a learnable direction.
#[derive(Module, Debug)]
pub struct WNConv2d<B: Backend> {
    /// Direction parameter `v`  — shape `[out_ch, in_ch/groups, kH, kW]`
    pub weight_v: Param<Tensor<B, 4>>,
    /// Magnitude parameter `g` — shape `[out_ch, 1, 1, 1]`
    pub weight_g: Param<Tensor<B, 4>>,
    /// Optional bias — shape `[out_ch]`
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub stride: [usize; 2],
    pub kernel_size: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    pub padding: Ignored<PaddingConfig2d>,
}

impl<B: Backend> WNConv2d<B> {
    /// Forward pass.
    ///
    /// # Shapes
    /// - `input`:  `[batch, channels_in,  height_in,  width_in]`
    /// - `output`: `[batch, channels_out, height_out, width_out]`
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // ── weight normalisation ─────────────────────────────────────────────
        // ‖v‖  (per output-channel L2 norm, kept as [out, 1, 1, 1])
        let v = self.weight_v.val();
        let g = self.weight_g.val();

        let v_norm = v
            .clone()
            .powf_scalar(2.0_f32)
            .sum_dim(1)
            .sum_dim(1)
            .sum_dim(1)
            .sqrt()
            .reshape([v.dims()[0], 1, 1, 1]); // [out, 1, 1, 1]

        // w = g * v / ‖v‖
        let weight = v * g / (v_norm + 1e-8_f32);

        // ── padding ──────────────────────────────────────────────────────────
        let [_batch, _ch_in, height_in, width_in] = input.dims();
        let ((top, bottom), (left, right)) = Self::calculate_padding_2d_pairs(
            (*self.padding).clone(),
            height_in,
            width_in,
            &self.kernel_size,
            &self.stride,
        );

        let options = PaddedConvOptions::asymmetric(
            self.stride,
            [top, left],
            [bottom, right],
            self.dilation,
            self.groups,
        );

        conv2d(input, weight, self.bias.as_ref().map(|b| b.val()), options)
    }

    fn calculate_padding_2d_pairs(
        padding: PaddingConfig2d,
        height: usize,
        width: usize,
        kernel_size: &[usize; 2],
        stride: &[usize; 2],
    ) -> ((usize, usize), (usize, usize)) {
        match padding {
            PaddingConfig2d::Valid => ((0, 0), (0, 0)),
            PaddingConfig2d::Same => {
                let (top, bottom) = Self::calculate_same_padding(kernel_size[0], stride[0], height);
                let (left, right) = Self::calculate_same_padding(kernel_size[1], stride[1], width);
                ((top, bottom), (left, right))
            }
            PaddingConfig2d::Explicit(top, left, bottom, right) => ((top, bottom), (left, right)),
        }
    }

    fn calculate_same_padding(kernel_size: usize, stride: usize, size_in: usize) -> (usize, usize) {
        let size_out = size_in.div_ceil(stride); // ceil division for same padding
        let total_padding = if size_out > 0 {
            let needed = (size_out - 1) * stride + kernel_size;
            needed.saturating_sub(size_in)
        } else {
            0
        };
        let pad_start = total_padding / 2;
        let pad_end = total_padding - pad_start;
        (pad_start, pad_end)
    }
}

#[derive(Module, Debug)]
pub struct SNConv2d<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Option<Param<Tensor<B, 1>>>,
}

#[derive(Module, Clone, Debug)]
pub struct Flip {}

impl Flip {
    #[trace_shapes]
    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>, reverse: bool) -> (Tensor<B, 3>, f32) {
        (x.flip([1]), 0.0)
    }
}

#[derive(Module, Debug, Clone)]
pub struct Log {}

impl Log {
    #[trace_shapes]
    pub fn forward<B: Backend>(
        &self,
        x: Tensor<B, 3>,
        x_mask: Tensor<B, 3>,
        reverse: bool,
    ) -> (Tensor<B, 3>, f32) {
        if !reverse {
            let y = x.clamp_min(1e-5).log() * x_mask;
            let logdet = y
                .clone()
                .neg()
                .sum_dims(&[1, 2])
                .to_data()
                .to_vec()
                .unwrap()[0];
            (y, logdet)
        } else {
            let x = x.exp() * x_mask;
            (x, 0.0)
        }
    }
}

const DEFAULT_MIN_BIN_WIDTH: f64 = 1e-3;
const DEFAULT_MIN_BIN_HEIGHT: f64 = 1e-3;
const DEFAULT_MIN_DERIVATIVE: f64 = 1e-3;

fn piecewise_rational_quadratic_transform<B: Backend>(
    inputs: Tensor<B, 3>,
    unnormalized_widths: Tensor<B, 4>,
    unnormalized_heights: Tensor<B, 4>,
    unnormalized_derivatives: Tensor<B, 4>,
    inverse: bool,
    tails: Option<&str>,
    tail_bound: f64,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let min_bin_width = DEFAULT_MIN_BIN_WIDTH;
    let min_bin_height = DEFAULT_MIN_BIN_HEIGHT;
    let min_derivative = DEFAULT_MIN_DERIVATIVE;

    if tails.is_none() {
        let (t_ret_1, t_ret_2) = rational_quadratic_spline(
            inputs.squeeze(),
            unnormalized_widths.squeeze(),
            unnormalized_heights.squeeze(),
            unnormalized_derivatives.squeeze(),
            inverse,
            0.0,
            1.0,
            0.0,
            1.0,
            min_bin_width,
            min_bin_height,
            min_derivative,
        );
        (t_ret_1.unsqueeze(), t_ret_2.unsqueeze())
    } else {
        unconstrained_rational_quadratic_spline(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse,
            min_bin_width,
            min_bin_height,
            min_derivative,
            tails,
            tail_bound,
        )
    }
}

fn unconstrained_rational_quadratic_spline<B: Backend>(
    inputs: Tensor<B, 3>,
    unnormalized_widths: Tensor<B, 4>,
    unnormalized_heights: Tensor<B, 4>,
    mut unnormalized_derivatives: Tensor<B, 4>,
    inverse: bool,
    min_bin_width: f64,
    min_bin_height: f64,
    min_derivative: f64,
    tails: Option<&str>,
    tail_bound: f64,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let inside_interval_mask = inputs
        .clone()
        .greater_equal_elem(-tail_bound)
        .bool_and(inputs.clone().lower_equal_elem(tail_bound));
    let outside_interval_mask = inside_interval_mask.clone().bool_not();

    let mut outputs = Tensor::zeros_like(&inputs);
    let mut logabsdet = Tensor::zeros_like(&inputs);

    if Some("linear") == tails {
        unnormalized_derivatives =
            unnormalized_derivatives.pad([(0, 0), (0, 0), (0, 0), (1, 1)], PadMode::Constant(0.0));
        let constant = ((1.0 - min_derivative).exp() - 1.0).ln();
        unnormalized_derivatives =
            unnormalized_derivatives.slice_fill([s![..], s![..], s![..], s![0]], constant);
        unnormalized_derivatives =
            unnormalized_derivatives.slice_fill([s![..], s![..], s![..], s![-1]], constant);
        outputs = outputs.mask_where(outside_interval_mask.clone(), inputs.clone());
        logabsdet = logabsdet.clone().mask_fill(outside_interval_mask, 0);
    } else {
        panic!("{:?} tails are not implemented.", tails);
    }

    let inputs = inputs
        .clone()
        .mask_where(inside_interval_mask.clone(), inputs)
        .squeeze::<1>();

    let unnormalized_widths = unnormalized_widths.clone().mask_where(
        inside_interval_mask.clone().unsqueeze_dim(3),
        unnormalized_widths,
    );
    let unnormalized_heights = unnormalized_heights.clone().mask_where(
        inside_interval_mask.clone().unsqueeze_dim(3),
        unnormalized_heights,
    );
    let unnormalized_derivatives = unnormalized_derivatives.clone().mask_where(
        inside_interval_mask.clone().unsqueeze_dim(3),
        unnormalized_derivatives,
    );

    let (t_outputs, t_logabsdet) = rational_quadratic_spline(
        inputs,
        unnormalized_widths.squeeze(),
        unnormalized_heights.squeeze(),
        unnormalized_derivatives.squeeze(),
        inverse,
        -tail_bound,
        tail_bound,
        -tail_bound,
        tail_bound,
        min_bin_width,
        min_bin_height,
        min_derivative,
    );

    let outputs = outputs.mask_where(inside_interval_mask.clone(), t_outputs.unsqueeze::<3>());
    let logabsdet =
        logabsdet.mask_where(inside_interval_mask.clone(), t_logabsdet.unsqueeze::<3>());

    (
        outputs
            .clone()
            .mask_where(inside_interval_mask.clone(), outputs),
        logabsdet
            .clone()
            .mask_where(inside_interval_mask, logabsdet),
    )
}

fn rational_quadratic_spline<B: Backend>(
    inputs: Tensor<B, 1>,
    unnormalized_widths: Tensor<B, 2>,
    unnormalized_heights: Tensor<B, 2>,
    unnormalized_derivatives: Tensor<B, 2>,
    inverse: bool,
    left: f64,
    right: f64,
    bottom: f64,
    top: f64,
    min_bin_width: f64,
    min_bin_height: f64,
    min_derivative: f64,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let num_bins = (*unnormalized_widths.dims().iter().last().unwrap()) as f64;
    let widths = softmax(
        unnormalized_widths.clone(),
        unnormalized_widths.dims().len() - 1,
    );
    let widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths;
    let cumwidths = widths.clone().cumsum(widths.dims().len() - 1);
    let cumwidths = cumwidths.pad([(0, 0), (1, 0)], PadMode::Constant(0.0));
    let cumwidths = (right - left) * cumwidths + left;
    let cumwidths = cumwidths.slice_fill([s![..], s![0]], left);
    let cumwidths = cumwidths.slice_fill([s![..], s![-1]], right);
    let widths =
        cumwidths.clone().slice([s![..], s![1..]]) - cumwidths.clone().slice([s![..], s![..-1]]);
    let derivatives = min_derivative + softplus(unnormalized_derivatives, 1.0);
    let heights = softmax(
        unnormalized_heights.clone(),
        unnormalized_heights.dims().len() - 1,
    );
    let heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights;
    let cumheights = heights.clone().cumsum(heights.dims().len() - 1);
    let cumheights = cumheights.pad([(0, 0), (1, 0)], PadMode::Constant(0.0));
    let cumheights = (top - bottom) * cumheights + bottom;
    let cumheights = cumheights.slice_fill([s![..], s![0]], bottom);
    let cumheights = cumheights.slice_fill([s![..], s![-1]], top);
    let heights =
        cumheights.clone().slice([s![..], s![1..]]) - cumheights.clone().slice([s![..], s![..-1]]);

    let bin_idx = if inverse {
        searchsorted(cumheights.clone(), inputs.clone(), 1e-6)
    } else {
        searchsorted(cumwidths.clone(), inputs.clone(), 1e-6)
    };

    let bin_idx: Tensor<B, 2, Int> = bin_idx.unsqueeze_dim::<2>(1);

    let input_cumwidths = cumwidths
        .clone()
        .gather(cumwidths.dims().len() - 1, bin_idx.clone())
        .slice([s![..], s![0]])
        .squeeze::<1>();

    let input_bin_widths = widths
        .clone()
        .gather(widths.dims().len() - 1, bin_idx.clone())
        .slice([s![..], s![0]])
        .squeeze::<1>();

    let input_cumheights = cumheights
        .clone()
        .gather(cumheights.dims().len() - 1, bin_idx.clone())
        .slice([s![..], s![0]])
        .squeeze::<1>();

    let delta = heights.clone() / widths;
    let input_delta = delta
        .clone()
        .gather(delta.dims().len() - 1, bin_idx.clone())
        .slice([s![..], s![0]])
        .squeeze::<1>();

    let input_derivatives = derivatives
        .clone()
        .gather(derivatives.dims().len() - 1, bin_idx.clone())
        .slice([s![..], s![0]])
        .squeeze::<1>();

    let input_derivatives_plus_one = derivatives
        .clone()
        .slice([s![..], s![1..]])
        .gather(derivatives.dims().len() - 1, bin_idx.clone())
        .slice([s![..], s![0]])
        .squeeze::<1>();

    let input_heights = heights
        .clone()
        .gather(heights.dims().len() - 1, bin_idx)
        .slice([s![..], s![0]])
        .squeeze::<1>();

    if inverse {
        let a = (inputs.clone() - input_cumheights.clone())
            * (input_derivatives.clone() + input_derivatives_plus_one.clone()
                - 2 * input_delta.clone())
            + input_heights.clone() * (input_delta.clone() - input_derivatives.clone());
        let b: Tensor<B, 1> = input_heights * input_derivatives.clone()
            - (inputs.clone() - input_cumheights.clone())
                * (input_derivatives.clone() + input_derivatives_plus_one.clone()
                    - 2 * input_delta.clone());
        let c = -input_delta.clone() * (inputs - input_cumheights.clone());

        let discriminant: Tensor<B, 1> = b.clone().powi_scalar(2) - 4 * a * c.clone();

        assert!(
            discriminant
                .clone()
                .greater_equal_elem(0.0)
                .all()
                .to_data()
                .to_vec::<bool>()
                .unwrap()[0]
        );

        let root: Tensor<B, 1> = (2 * c) / (-b - discriminant.sqrt());

        let outputs = root.clone() * input_bin_widths + input_cumwidths;

        let theta_one_minus_theta: Tensor<B, 1> = root.clone() * (1.0 - root.clone());

        let denominator: Tensor<B, 1> = input_delta.clone()
            + ((input_derivatives.clone() + input_derivatives_plus_one.clone()
                - 2 * input_delta.clone())
                * theta_one_minus_theta.clone());
        let t_root: Tensor<B, 1> = 1.0 - root.clone();
        let derivative_numerator: Tensor<B, 1> = input_delta.clone().powi_scalar(2)
            * (input_derivatives_plus_one * root.powi_scalar(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (t_root).powi_scalar(2));

        let logabsdet: Tensor<B, 1> = derivative_numerator.log() - 2.0 * denominator.log();

        return (outputs, -logabsdet);
    }

    let theta = (inputs - input_cumwidths) / input_bin_widths;
    let theta_one_minus_theta: Tensor<B, 1> = theta.clone() * (1 - theta.clone());
    let numerator = input_heights
        * (input_delta.clone() * theta.clone().powi_scalar(2)
            + input_derivatives.clone() * theta_one_minus_theta.clone());
    let denominator: Tensor<B, 1> = input_delta.clone()
        + ((input_derivatives.clone() + input_derivatives_plus_one.clone()
            - 2 * input_delta.clone())
            * theta_one_minus_theta.clone());
    let outputs = input_cumheights + numerator / denominator.clone();

    let t_theta: Tensor<B, 1> = 1 - theta.clone();
    let derivative_numerator: Tensor<B, 1> = input_delta.clone().powi_scalar(2)
        * (input_derivatives_plus_one * theta.powi_scalar(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (t_theta).powi_scalar(2));

    let logabsdet: Tensor<B, 1> = derivative_numerator.log() - 2.0 * denominator.log();

    return (outputs, -logabsdet);
}

fn searchsorted<B: Backend>(
    bin_locations: Tensor<B, 2>,
    inputs: Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, 1, Int> {
    let bin_locations = bin_locations.clone().slice_assign(
        [s![..], s![-1]],
        bin_locations.slice([s![..], s![-1]]) + eps,
    );

    inputs
        .unsqueeze_dim(1)
        .greater_equal(bin_locations)
        .int()
        .sum_dims_squeeze(&[-1])
        - 1
}
