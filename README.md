# LLaMA-7B Frontend

This is a **non-production ready** frontend for [LLaMA-7B](https://github.com/facebookresearch/llama). Do not expose this to the internet unless you are prepared to have random scanners instantly take control of your machine.

![Demonstration video](demo.gif)

## Requirements

- Linux (not sure if this works on Windows)
- Python 3.10+, pip
- NVidia (?) GPU with 16GB+ of VRAM

## Setup

1. Clone this repo.
2. `pip install -r requirements.txt`
3. `pip install -e .`
4. Copy `7B` checkpoints folder and `tokenizer.model` to repository root.

## Running

`flask run` will start the app server on `127.0.0.1:5000`. Enable access from other devices on the network with `flask run --host 0.0.0.0`.

## Configuration

The checkpoint folder and tokenizer file can be configured with environment variables.

| Variable     | Description     | Default value |
|--------------|-----------|------------|
| CHECKPOINT_DIR | Path of 7B checkpoint folder      | `7B`        |
| TOKENIZER_PATH      | Path of tokenizer model  | `tokenizer.model`       |