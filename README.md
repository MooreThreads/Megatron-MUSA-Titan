# Megatron-LM-MUSA-Patch


## Installation
You can create a directory named `megatron_dev,` and use the command below to clone the `Megatron-LM`, `megatron-lm-musa-patch`, `apex`, `TransformerEngine`, `flash-attention` to the `megatron_dev`.  
In the kuae release image, `apex`, `TransformerEngine`, `flash-attention` is already installed, you can skip the installation of three repos above.

```bash
# Megatron-LM
git clone https://sh-code.mthreads.com/ai/Megatron-LM.git
pushd Megatron-LM
git checkout -b core_r0.9.0 core_r0.9.0
popd

# megatron-lm-musa-patch
git clone https://sh-code.mthreads.com/ai/megatron-lm-musa-patch.git
pushd megatron-lm-musa-patch
git fetch origin dev
git checkout -b dev origin/dev
popd

# apex (optional)
git clone https://sh-code.mthreads.com/ai/apex
pushd apex
git fetch origin feature/kuae_1.2
git checkout -b feature/kuae_1.2 origin/feature/kuae_1.2
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd

# TransformerEngine (optional)
git clone https://sh-code.mthreads.com/ai/TransformerEngine.git
pushd TransformerEngine
git fetch origin lj_fp8
git checkout -b lj_fp8 origin/lj_fp8
bash install.sh
popd

# flash-attention (optional)
git clone https://sh-code.mthreads.com/ai/flash-attention.git
pushd flash-attention
git fetch origin musa_dev
git checkout -b musa_dev origin/musa_dev
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE python setup.py develop
popd

```

## Getting started
### Llama3 

```bash
cd megatron-lm-musa-patch/examples/llama3
bash dist_run_pretrain_megatron_llama3_musa.sh
```

### Mixtral

```bash
cd megatron-lm-musa-patch/examples/mixtral
bash dist_run_pretrain_megatron_llama3_musa.sh
```

### Llava

```bash
cd megatron-lm-musa-patch/examples/llava

```

### DeepSeekV3

```bash
cd megatron-lm-musa-patch/examples/deepseekv3

```