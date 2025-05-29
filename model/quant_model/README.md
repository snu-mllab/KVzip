## Implementing QServe Quantized Model
This implementation borrows code from [DuoAttention](https://github.com/mit-han-lab/duo-attention).

- git submodule update --init --recursive
- pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
- cd omniserve
- pip install .
- cd kernel
- pip install .

Then, set `-m llama3-8b-4m-w8a8kv4`.

