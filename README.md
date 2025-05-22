## PASA: online pseudo-average shifting Attention for Long-sequence LLM inference

### Introduction
PASA algorithm is developed for accelerating attention calculation using fully low-precision computing like fp16. It is numerically robust and will not suffer from the numerical stability like overflow and large numerical error caused by the large rounding error. It is quite effective for memory-restricted and vector-computing-power-restricted AI architectures like NPU. 

### Verification Platform
**[Hardware]**: Ascend NPU (Atlas 800I - A2)
**[Software]**: CANN 8.0.0 and Torch-NPU

### Installation
> Step 1: install CANN driver and ops libraries
```


```

> Step 2: Install pytorch and torch\_npu in conda
```
pip install pytorch==2.1.0
pip install torch_npu==2.1.0
```

> Step 3: run test cases
The test cases are divided into two categorites: random dataset and real LMs. The random datasets can be simulated in a separate manner with PASA, while PASA must be integrated into real LM scripts. Hence, the following procedures are only related to the random datasets.

In ```PASA_Verification.py```, there are two parameters controlling the generated dataset: ```format_flag``` and ```sub_case_no```. We set ```format_flag = 1``` representing the random datasets. For ```sub_case_no```:

```
sub_case_no = 0   # represent random dataset with uniform distritbuion
```
```
sub_case_no = 1   # represent random dataset with normal + Bernoulli hybrid distribution in FA3.0
```

In addition, the ```mean_val``` controls the mean value of the generated data 





