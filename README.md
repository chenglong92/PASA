## PASA: online pseudo-average shifting Attention for Long-sequence LLM inference

### 1. Introduction
PASA algorithm is developed for accelerating attention calculation using fully low-precision computing like fp16. It is numerically robust and will not suffer from the numerical stability like overflow and large numerical error caused by the large rounding error. It is quite effective for memory-restricted and vector-computing-power-restricted AI architectures like NPU. 

### 2. Verification Platform
**[Hardware]**: Ascend NPU (Atlas 800I - A2)  
**[Software]**: CANN 8.0.0 and Torch-NPU

### 3. Installation
> Step 1: install CANN driver and ops libraries
```bash
wget --no-check-certificate https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-toolkit_8.0.0_linux-aarch64.run

wget --no-check-certificate https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-kernels-910_8.0.0_linux-aarch64.run
```
```bash
chmod +x Ascend-cann-toolkit_8.0.0_linux-aarch64.run
./Ascend-cann-toolkit_8.0.0_linux-aarch64.run
chmod +x Ascend-cann-kernels-910_8.0.0_linux-aarch64.run
./Ascend-cann-kernels-910_8.0.0_linux-aarch64.run
```
```
source $ASCEND_INSTALL_PATH/ascend-toolkit/set_env.sh
```

> Step 2: Install pytorch and torch\_npu in conda
```bash
conda create -n env_name python=3.9
conda activate env_name
```
```
conda install pyyaml setuptools wheel typing_extensions numpy protobuf attrs pathlib2 scipy requests psutil absl-py decorator
```

```bash
pip install pytorch==2.1.0
pip install torch_npu==2.1.0
```

> Step 3: run test cases
The test cases are divided into two categorites: random dataset and real LMs. The random datasets can be simulated in a separate manner with PASA, while PASA must be integrated into real LM scripts. Hence, the following procedures are only related to the random datasets.

In ```PASA_Verification.py```, there are two parameters controlling the generated dataset: ```format_flag``` and ```sub_case_no```. We set ```format_flag = 1``` representing the random datasets. For ```sub_case_no```:

```python
sub_case_no = 0   # represent random dataset with uniform distritbuion
```
```python
sub_case_no = 1   # represent random dataset with normal + Bernoulli hybrid distribution in FA3.0
```

In addition, the ```mean_val``` controls the mean value of the generated data, while the ```Am`` controls the amplitude of the generated datases. In PASA's paper, the candidates are:

| ```mean_val``` | ```Am``` |
| :------: | :------: |
|  $0, 10, 20, 30$ | $0.5, 5, 10, 20$ |


after the parameter setup, just run: 
```bash
python PASA_Verification.py
```

### 4. optimal parater calculation
Just run:
```
python optimal_para.py
```
In the ```PASA_Verification.py```, the optimal parameter is fixed to $afa = 0.9689939$.
