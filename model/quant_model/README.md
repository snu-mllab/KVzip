## Implementing QServe Quantized Model
This implementation borrows code from [DuoAttention](https://github.com/mit-han-lab/duo-attention).

```bash
git submodule update --init --recursive
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
cd model/quant_model
pip install .
cd omniserve
pip install .
cd kernel
pip install .
```

Then, go to root directory and set `-m llama3-8b-4m-w8a8kv4` to run the code.
For example,
- `python -B test.py -m llama3-8b-4m-w8a8kv4 -d squad -r 0.3 --kv_type retain`

Note, the evict cache is currently not available for QServe quantized models.

