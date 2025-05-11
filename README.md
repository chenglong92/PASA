# PASA: online pseudo-average shifting Attention

PASA algorithm is developed for accelerating attention calculation using fully low-precision computing like fp16. It is numerically robust and will not suffer from the numerical stability like overflow and large numerical error caused by the large rounding error. It is quite effective for memory-restricted and vector-computing-power-restricted AI architectures like NPU. 

**[Hardware]**: Ascend NPU
**[Software]**: CANN 8.0.0 and Torch-NPU


