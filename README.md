# How to use

Implementation of piper/vits tts using [burn](https://github.com/tracel-ai/burn). Only inference is supported for now.

* extract the state_dict from pytorch model, this will create `<model>_state_dict.ckpt`
```
 uv venv && uv pip install torch
 uv run scripts/extract_state_dict.py <path to ckpt>
```
* convert the weights to burnpack fromat
```
cargo run --release convert \
    --model-path <path to model_state_dict.ckpt> \
    --model-config <path to model config json> \
    --model-quality <low, medium, high> 
    --model-name <voice name> \
    --output-path models/
```
* run the model
```
cargo run --release run \
    --model-path models/<model name>.bpk \
    --model-config models/<model name>.json \
    --target-text "Hello piper"
```
