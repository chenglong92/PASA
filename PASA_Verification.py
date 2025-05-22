import torch
import functools
import torch_npu

import math
import numpy as np
import os
import random

from torch.library import Library, impl
from compare import dataCompare

def gendata(batch_size, num_heads, num_kv_heads, seq_len1, seq_len2, head_size, mean_val = 0, Am = 0.5, distype = 0, gendatatype = torch.float32, datatype = torch.float16, seed = 2024724):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (distype == 0):
        # uniform random distribution
        #Am = 1.0
        q = (torch.rand(batch_size, num_heads, seq_len1, head_size, dtype=gendatatype) - 0.5) * 2.0 * Am + mean_val
        k = (torch.rand(batch_size, num_kv_heads, seq_len2, head_size, dtype=gendatatype) - 0.5) * 2.0 * Am + mean_val
        v = (torch.rand(batch_size, num_kv_heads, seq_len2, head_size, dtype=gendatatype) - 0.5) * 2.0 * Am + mean_val
    if (distype == 1):
        # hybrid random distribution(normal + Bernoulli)
        #Am = 10.0
        q = torch.normal(mean_val, 1, size=(batch_size, num_heads, seq_len1, head_size), dtype = gendatatype)
        q += torch.normal(0, Am, size=(batch_size, num_heads, seq_len1, head_size), dtype = gendatatype) * np.random.binomial(size=(batch_size, num_heads, seq_len1, head_size), n = 1, p = 0.001)
        k = torch.normal(mean_val, 1, size=(batch_size, num_kv_heads, seq_len2, head_size), dtype = gendatatype)
        k += torch.normal(0, Am, size=(batch_size, num_kv_heads, seq_len2, head_size), dtype = gendatatype) * np.random.binomial(size=(batch_size, num_kv_heads, seq_len2, head_size), n = 1, p = 0.001)
        v = torch.normal(mean_val, 1, size=(batch_size, num_kv_heads, seq_len2, head_size), dtype = gendatatype)
        v += torch.normal(0, Am, size=(batch_size, num_kv_heads, seq_len2, head_size), dtype = gendatatype) * np.random.binomial(size=(batch_size, num_kv_heads, seq_len2, head_size), n = 1, p = 0.001)
        #
    q = q.type(datatype)
    k = k.type(datatype)
    v = v.type(datatype)
    return q, k, v

def softmax(x, is_fp16=False):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    return y, x_max, x_sum

def goldenFA(q, k, v, scalar, iB, iN_Q, iN_KV, OutputDtype = torch.float16, attention_mask=None):
    B, N, S, D = k.shape
    qq = q[iB, iN_Q, :, :]
    kk = k[iB, iN_KV, :, :]
    vv = v[iB, iN_KV, :, :]
    qq = qq.to(torch.float32).numpy()
    kk = kk.to(torch.float32).numpy()
    vv = vv.to(torch.float32).numpy()
    atten_matrix = qq @ kk.T * scalar
    if attention_mask != None:
        atten_mask1 = attention_mask[iB, :, :]
        atten_matrix += atten_mask1.to(torch.float32).numpy() * (-10000)
    p, m, l = softmax(atten_matrix)
    #
    p = p / np.tile(l, S)
    bmm2 = p @ vv
    return torch.from_numpy(bmm2).to(OutputDtype)

class tiling_par:
    s1 = int(128)
    s2 = int(128)
    d = int(64)
    afa = float(0.99)
    M = None
    def __init__(self, s1, s2, d, afa):
        self.s1 = s1
        self.s2 = s2
        self.d = d
        self.afa = afa
    def constructM(self, n):
        self.M = torch.eye(n) - torch.ones((n, n)) * self.afa / n

def rowmean(x):
    S_ave = np.mean(x, axis = 1, keepdims=True)
    return S_ave

def RMSE(x, golden):
    x1 = x.astype(np.float64)
    golden1 = golden.astype(np.float64)
    z = (x1 - golden1)**2
    RSE = np.sum(z)
    RMSE = np.sqrt(RSE / x.size)
    #
    z = golden1**2
    RSE = np.sum(z)
    RMSE_golden = np.sqrt(RSE / x1.size)
    RMSE_Re = RMSE / RMSE_golden
    return RMSE, RMSE_Re

def RMSE_ab(x):
    x1 = x.astype(np.float64)
    z = (x1)**2
    RSE = np.sum(z)
    RMSE = np.sqrt(RSE / x.size)
    return RMSE

def obtainInvPam(M):
    b = np.float32(-M[0,1])
    a = np.float32(M[0,0]) + b
    s2 = M.shape[0]
    Inv_Pam = b * s2 / (a * (a - b * s2))
    return Inv_Pam

def softmax_npu(x):
    x_max = x.max(axis=-1, keepdim=True)[0]
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdim=True)
    ans = y
    return ans, x_max, x_sum

def pasa(q_npu, k_npu, v_npu, scalar, Tiling_Para, iB=0, iN_Q=0, iN_KV=0, mm_tp=torch.float16, vec_tp=torch.float16, OutputDtype = torch.float16, attention_mask=None):
    device_id = q_npu.device
    # if q_npu.shape[2] == 1:
    #     q_npu = q_npu.repeat(1, 1, 1 ,1)
    q = q_npu.to(torch.float32).to(mm_tp)
    k = k_npu.to(torch.float32).to(mm_tp)
    v = v_npu.to(torch.float32).to(mm_tp)
    B, num_heads, S1, D = q.shape
    B, num_kv_heads, S2, D = k.shape
    N = num_heads
    k = torch.repeat_interleave(k, repeats=(num_heads // num_kv_heads), dim=1)
    v = torch.repeat_interleave(v, repeats=(num_heads // num_kv_heads), dim=1)
    s1 = Tiling_Para.s1
    s2 = Tiling_Para.s2
    actual_S2 = (((S2 - 1) // s2) + 1) * s2
    Padding_LenKV = actual_S2 - S2
    if Padding_LenKV != 0:
        if S2 < s2:
            Padding_K_Zeros = torch.zeros((B, N, Padding_LenKV, D), dtype=vec_tp).to(device_id)
            Padding_V_Zeros = torch.zeros((B, N, Padding_LenKV, D), dtype=vec_tp).to(device_id)
        else:
            # why
            Padding_K_Zeros = k[:, :, -Padding_LenKV:, :]
            Padding_V_Zeros = v[:, :, -Padding_LenKV:, :]
        k = torch.cat((k, Padding_K_Zeros), dim = (k.dim() -2))
        v = torch.cat((v, Padding_V_Zeros), dim = (v.dim() -2))
        if attention_mask == None:
            attention_mask = torch.zeros((B, N, S1, S2), dtype=vec_tp).to(device_id)
        attention_mask_ones = torch.ones((B, N, S1, Padding_LenKV), dtype=vec_tp).to(attention_mask.dtype).to(device_id)
        attention_mask = torch.cat((attention_mask, attention_mask_ones), dim=(attention_mask.dim() - 1))
    S2 = actual_S2
    if attention_mask != None:
        attention_mask1 = attention_mask[:, :, :, :].type(vec_tp) * (-torch.inf)
        attention_mask1 = torch.nan_to_num(attention_mask1, nan=0.0)
    Nq = int(S1 / s1)
    Nkv = int(S2 / s2)
    Padding_LenKV = S2 - Nkv * s2
    Padding_LenQ = S1 - Nq * s1
    if Padding_LenKV != 0:
        raise Exception("Padding_LebKV should be zero")
    #
    Loop_Q = Nq
    Loop_KV = Nkv
    if Padding_LenQ != 0:
        Loop_Q = Nq + 1
    Inv_Pam = Tiling_Para.afa / (1.0 - Tiling_Para.afa)
    Inv_Pam = Inv_Pam.to(vec_tp).to(k.device)
    M = Tiling_Para.M
    M = M * scalar
    M = M.to(mm_tp).to(device_id)
    #
    O = torch.zeros((B, N, S1, D), dtype=vec_tp).to(device_id)
    actual_s1 = Tiling_Para.s1
    actual_s2 = Tiling_Para.s2
    for iNq in range(Loop_Q):
        if iNq == Nq:
            actual_s1 = Padding_LenQ
        m_j_1 = torch.zeros((B, N, actual_s1, 1), dtype=vec_tp, device=k.device) - 60000
        l_j_1 = torch.zeros((B, N, actual_s1, 1), dtype=vec_tp, device=k.device)
        ave_j_1 = torch.zeros((B, N, actual_s1, 1), dtype=vec_tp, device=k.device)
        O0 = torch.zeros((B, N, actual_s1, D), dtype=vec_tp, device=k.device)
        for iNkv in range(Loop_KV):
            if (iNq == 0):
                # tensor转置
                temp_K = k[:, :, (iNkv*s2):(iNkv*s2+s2), :]
                temp_K = temp_K.permute(0, 1, 3, 2) @ M
                k[:, :, (iNkv*s2):(iNkv*s2+s2), :] = temp_K.permute(0, 1, 3, 2)
            q0 = q[:, :, (iNq*s1):(iNq*s1+s1), :]
            k0 = k[:, :, (iNkv*s2):(iNkv*s2+s2), :]
            # q0 = q0.numpy().astype(np.float16)
            # k0 = k0.permute(0, 1, 3, 2).numpy().astype(np.float16)
            # S = np.matmul(q0, k0)
            S = torch.matmul(q0,(k0.permute(0, 1, 3, 2)))
            # S = torch.from_numpy(S)
            #print(S)
            if (vec_tp == torch.float32):
                # using fp16-fp32 hybrid-precision PASA.
                S = S.float()
            #obtain row mean.
            S_ave = torch.mean(S,axis=-1,keepdim=True)
            #
            if attention_mask != None:
                if q_npu.shape[2] == 1:
                    local_attention_mask = attention_mask1[:, :, -1, (iNkv*s2):(iNkv*s2+s2)].unsqueeze(2)
                else:
                    local_attention_mask = attention_mask1[:, :, (iNq*s1):(iNq*s1+s1), (iNkv*s2):(iNkv*s2+s2)]
                # print(S.shape)
                # print(local_attention_mask[:, :, -1, :].unsqueeze(2))
                # if q.shape[2] == 1:
                #     import pdb; pdb.set_trace()
                S += local_attention_mask[:, :, -1, :].unsqueeze(2) if q_npu.shape[2] == 1 else local_attention_mask
                # S += local_attention_mask

            p, m_j, l_j = softmax_npu(S)
            #
            if iNkv == 0:
                S_ave_0 = S_ave
            S_ave_temp = S_ave_0 * 1.0
            Temp_Ratio1 = 1.0 / (iNkv + 1)
            Temp_Ratio2 = iNkv / (iNkv + 1)
            Temp_Ratio1 = torch.tensor(Temp_Ratio1, dtype = vec_tp)
            Temp_Ratio2 = torch.tensor(Temp_Ratio2, dtype = vec_tp)
            S_ave_0 = S_ave * Temp_Ratio1 + S_ave_0 * Temp_Ratio2
            ave_j = (S_ave-S_ave_0) * Inv_Pam
            ave_j_1 = Inv_Pam * (S_ave_temp - S_ave_0)
            m_temp = m_j + ave_j
            #
            m_temp0 = m_j_1 + Inv_Pam * (S_ave_temp - S_ave_0)
            m_temp = torch.max(torch.cat((m_temp, m_temp0), -1), axis=-1, keepdim=True)[0]
            if (iNkv == 0):
                l_j_1 = torch.exp(m_j - m_temp + ave_j_1) * l_j
            else:
                l_j_1 = torch.exp(m_j_1 - m_temp + ave_j_1) * l_j_1 + torch.exp(m_j - m_temp + ave_j) * l_j
            temp_v = v[:, :, (iNkv*s2):(iNkv*s2+s2), :]
            temp_O = torch.matmul(p, temp_v)
            #print(temp_O)
            if (iNkv == 0):
                O0 = torch.tile(torch.exp(m_j - m_temp+ave_j_1), (1, 1, 1, D)) * temp_O
            else:
                O0 = torch.tile(torch.exp(m_j_1 - m_temp+ave_j_1), (1, 1, 1, D)) * O0 + torch.tile(torch.exp(m_j - m_temp+ave_j), (1,1,1,D)) * temp_O
            m_j_1 = m_temp
        O[:,:,(iNq*s1):(iNq*s1+s1), :] = O0 / torch.tile(l_j_1, (1, 1, 1, D))
    return O.to(torch.float32).to(OutputDtype)

def getAttentionMask(batch_size, seq_len1, seq_len2, datatype = torch.float16, types = 0):
    if types == 0:
        # full matrix.
        attention_mask = torch.zeros((batch_size, seq_len1, seq_len2), dtype = datatype)
    elif types == 1:
        # lower-diagonal
        attention_mask = torch.ones((batch_size, seq_len1, seq_len2), dtype = datatype)
        attention_mask = torch.triu(attention_mask)
    return attention_mask

if __name__ == "__main__":
    # load test data from npu device.
    batch_size = []
    seq_len1 = []
    seq_len2 = []
    num_heads = []
    num_kv_heads = []
    head_size = []
    InputDtype = []
    OutputDtype = []
    #
    # format_flag: 0 - real case(BNSD); 1 - generated random case(BNSD);  2 - BSH data
    # sub-case no: 1, 2, 3
    format_flag = 1
    sub_case_no = 0
    device_id = 6
    #
    if format_flag == 0:
        # load Q, K, V from the real network case.
        if (sub_case_no == 2):
            # X2-SVD-imag2vid
            print("TBA")
    if format_flag == 1:
        if sub_case_no == 0:
            # generated random case(uniform)
            batch_size = 1
            num_heads = 28
            num_kv_heads = 4
            seq_len1 = int(1280+127)
            seq_len2 = int(1280+127)
            head_size = 128
            #
            mean_val = 30.0
            Am = 0.5
            distype = int(0)
            #
            # input data: Q K V and output Q datatype: torch.float16 and torch.bfloat16
            InputDtype = torch.bfloat16
            OutputDtype = torch.bfloat16
            #
            attention_mask_stat = False
            #
            q_cpu, k_cpu, v_cpu = gendata(batch_size, num_heads, num_kv_heads, seq_len1, seq_len2, head_size, mean_val, Am, distype, torch.float32, InputDtype)
            q = q_cpu.npu(device_id)
            k = k_cpu.npu(device_id)
            v = v_cpu.npu(device_id)
        if sub_case_no == 1:
            # generated random case(Tri Dao's FA3.0)
            batch_size = 1
            num_heads = 28
            num_kv_heads = 4
            seq_len1 = int(1280+127)
            seq_len2 = int(1280+127)
            head_size = 128
            #
            mean_val = 30.0
            Am = 10.0
            distype = int(1)
            # input data: Q K V and output Q datatype: torch.float16 and torch.bfloat16
            InputDtype = torch.bfloat16
            OutputDtype = torch.bfloat16
            #
            attention_mask_stat = False
            #
            q_cpu, k_cpu, v_cpu = gendata(batch_size, num_heads, num_kv_heads, seq_len1, seq_len2, head_size, mean_val, Am, distype, torch.float32, InputDtype)
            q = q_cpu.npu(device_id)
            k = k_cpu.npu(device_id)
            v = v_cpu.npu(device_id)
    if format_flag == 2:
        # .
        print("TBA")
    #
    print(q.shape)
    print(k.shape)
    print(v.shape)
    #
    k1 = k[0:,0:,0:,0:] * 1.0
    #
    mm_tp=torch.float16
    vec_tp=torch.float16
    #
    baseM = int(128)
    baseN = int(128)
    DoublePassRatio = 0.005   # Five parts per thousand
    attention_mask = None
    #
    # init afa0
    if InputDtype == torch.float16:
        afa = torch.tensor(0.9375).float()
    elif InputDtype == torch.bfloat16:
        afa = torch.tensor(0.9689939).float()
    #
    Tiling_Para = tiling_par(baseM, baseN, head_size, afa)
    Tiling_Para.constructM(Tiling_Para.s2)
    scalar = (1 / math.sqrt(Tiling_Para.d))
    #
    # the results from PASA.
    compare_size = int(seq_len1 / Tiling_Para.s1) * Tiling_Para.s1
    #
    print("##################################################################################")
    print("                                PASA VS Standard Attention                        ")
    print("##################################################################################")
    O = np.zeros((batch_size, num_heads, seq_len1, head_size), dtype=np.float32)
    O_golden = np.zeros((batch_size, num_heads, seq_len1, head_size), dtype=np.float32)
    group_num = int(num_heads / num_kv_heads)
    RMSE_PASA_ALL = np.zeros((batch_size, num_heads), dtype=np.float32)
    RMSE_PASA_Re_ALL = np.zeros((batch_size, num_heads), dtype=np.float32)
    pass_res = np.zeros((batch_size, num_heads), dtype=np.str_)
    ltPct = np.zeros((batch_size, num_heads), dtype=np.float32)
    if attention_mask_stat == True:
        attention_mask = getAttentionMask(batch_size, seq_len1, seq_len2, torch.int8, 1)
    for iB in range(batch_size):
        for iN_KV in range(num_kv_heads):
            for i_q in range(group_num):
                i_q_head = iN_KV * group_num + i_q
                print(f"i_q_head:{i_q_head}")
                O[iB, i_q_head, :, :] = pasa(q, k, v, scalar, Tiling_Para, iB, i_q_head, iN_KV, mm_tp, vec_tp, OutputDtype, attention_mask).to(torch.float32).numpy()
                O_golden[iB, i_q_head, :, :] = goldenFA(q_cpu, k_cpu, v_cpu, scalar, iB, i_q_head, iN_KV, OutputDtype, attention_mask).to(torch.float32).numpy()
                pass_res[iB, i_q_head], ltPct[iB, i_q_head] = dataCompare(O[iB, i_q_head, :, :], O_golden[iB, i_q_head, :, :], DoublePassRatio, DoublePassRatio)
                RMSE_PASA, tmp = RMSE(O[iB, i_q_head, :, :], O_golden[iB, i_q_head, :, :])
                RMSE_PASA_ALL[iB, i_q_head] = RMSE_PASA
                RMSE_PASA_Re_ALL[iB, i_q_head] = RMSE_PASA / RMSE_ab(v_cpu[iB, iN_KV, :, :].to(torch.float32).numpy())
    pass_res_all, ltPct_all = dataCompare(O[:, :, :, :], O_golden[:, :, :, :], DoublePassRatio, DoublePassRatio)
    RMSE_PASA, tmp = RMSE(O, O_golden)
    RMSE_PASA_Re = RMSE_PASA / RMSE_ab(v_cpu.to(torch.float32).numpy())
    print(f"RMSE_PASA_all, RMSE_PASA_Re_all:{RMSE_PASA, RMSE_PASA_Re}")
    print(f"RMSE_PASA_Components:{RMSE_PASA_ALL}")
    print(f"RMSE_PASA_Re_Components:{RMSE_PASA_Re_ALL}")
    print(f"pass_res_all:{pass_res_all}, ltPct_all:{ltPct_all}")
    print(f"pass_res:{pass_res}")
    print(f"ltPct:{ltPct}")
    #
    # exit()
    print("##################################################################################")
    print("                        PFA on Torch_NPU(CANN) VS Standard Attention              ")
    print("##################################################################################")
    # the results from the PFA in CANN-torch
    attention_mask_npu = None
    if attention_mask_stat == True:
        attention_mask_npu = attention_mask.npu(device_id)
    q2 = q
    k2 = k1
    v2 = v
    npu_out, _ = torch_npu.npu_fused_infer_attention_score(
        q2, k2, v2, num_heads=num_heads, num_key_value_heads=num_kv_heads,
        input_layout='BNSD',
        pse_shift=None,
        sparse_mode=0,
        actual_seq_lengths = [seq_len1],
        actual_seq_lengths_kv = [seq_len2],
        pre_tokens = 65535,
        next_tokens = 65535,
        atten_mask=attention_mask_npu,
        scale=scalar,
        inner_precise=1
    )
    
    npu_out_hp, _ = torch_npu.npu_fused_infer_attention_score(
        q2, k2, v2, num_heads=num_heads, num_key_value_heads=num_kv_heads,
        input_layout='BNSD',
        pse_shift=None,
        sparse_mode=0,
        actual_seq_lengths = [seq_len1],
        actual_seq_lengths_kv = [seq_len2],
        pre_tokens = 65535,
        next_tokens = 65535,
        atten_mask=attention_mask_npu,
        scale=scalar,
        inner_precise=0
    )
    #
    print("================all heads: high-performance PFA(CANN)============================")
    group_num = int(num_heads / num_kv_heads)
    #O_golden = np.zeros((batch_size, num_heads, seq_len1, head_size), dtype=np.float16)
    RMSE_CANN_ALL = np.zeros((batch_size, num_heads), dtype=np.float32)
    RMSE_CANN_Re_ALL = np.zeros((batch_size, num_heads), dtype=np.float32)
    pass_res = np.zeros((batch_size, num_heads), dtype=np.str_)
    ltPct = np.zeros((batch_size, num_heads), dtype=np.float32)
    for iB in range(batch_size):
        for iN_KV in range(num_kv_heads):
            for i_q in range(group_num):
                i_q_head = iN_KV * group_num + i_q
                print(f"i_q_head:{i_q_head}")
                #O_golden[iB, i_q_head, :, :] = goldenFA(q_cpu, k_cpu, v_cpu, scalar, iB, i_q_head, iN_KV, attention_mask[iB, :, :])
                pass_res[iB, i_q_head], ltPct[iB, i_q_head] = dataCompare(npu_out[iB, i_q_head, :, :].to(torch.float32).cpu().numpy(), O_golden[iB, i_q_head, :, :], DoublePassRatio, DoublePassRatio)
                RMSE_CANN, tmp = RMSE(npu_out[iB, i_q_head, :, :].to(torch.float32).cpu().numpy(), O_golden[iB, i_q_head, :, :])
                RMSE_CANN_ALL[iB, i_q_head] = RMSE_CANN
                RMSE_CANN_Re_ALL[iB, i_q_head] = RMSE_CANN / RMSE_ab(v_cpu[iB, iN_KV, :, :].to(torch.float32).numpy())
    pass_res_all, ltPct_all = dataCompare(npu_out.to(torch.float32).cpu().numpy(), O_golden, DoublePassRatio, DoublePassRatio)
    PFA_CANN, tmp = RMSE(npu_out.to(torch.float32).cpu().numpy(), O_golden)
    RMSE_CANN_Re = PFA_CANN / RMSE_ab(v_cpu.to(torch.float32).numpy())
    print(f"PFA_CANN_hpf_all-RMSE, PFA_CANN_hpf_Re_all-RMSE:{PFA_CANN, RMSE_CANN_Re}")
    print(f"RMSE_CANN_Components:{RMSE_CANN_ALL}")
    print(f"RMSE_CANN_Re_Components:{RMSE_CANN_Re_ALL}")
    print(f"pass_res_all:{pass_res_all}, ltPct_all:{ltPct_all}")
    print(f"pass_res:{pass_res}")
    print(f"ltPct:{ltPct}")
    #'''
    #'''
    print("=================all heads: high-precision PFA(CANN)============================")
    group_num = int(num_heads / num_kv_heads)
    RMSE_CANN_ALL = np.zeros((batch_size, num_heads), dtype=np.float32)
    RMSE_CANN_Re_ALL = np.zeros((batch_size, num_heads), dtype=np.float32)
    pass_res = np.zeros((batch_size, num_heads), dtype=np.str_)
    ltPct = np.zeros((batch_size, num_heads), dtype=np.float32)
    for iB in range(batch_size):
        for iN_KV in range(num_kv_heads):
            for i_q in range(group_num):
                i_q_head = iN_KV * group_num + i_q
                print(f"i_q_head:{i_q_head}")
                pass_res[iB, i_q_head], ltPct[iB, i_q_head] = dataCompare(npu_out_hp[iB, i_q_head, :, :].to(torch.float32).cpu().numpy(), O_golden[iB, i_q_head, :, :], DoublePassRatio, DoublePassRatio)
                RMSE_CANN, tmp = RMSE(npu_out_hp[iB, i_q_head, :, :].to(torch.float32).cpu().numpy(), O_golden[iB, i_q_head, :, :])
                RMSE_CANN_ALL[iB, i_q_head] = RMSE_CANN
                RMSE_CANN_Re_ALL[iB, i_q_head] = RMSE_CANN / RMSE_ab(v_cpu[iB, iN_KV, :, :].to(torch.float32).numpy())
    pass_res_all, ltPct_all = dataCompare(npu_out_hp.to(torch.float32).cpu().numpy(), O_golden, DoublePassRatio, DoublePassRatio)
    PFA_CANN, tmp = RMSE(npu_out_hp.to(torch.float32).cpu().numpy(), O_golden)
    RMSE_CANN_Re = PFA_CANN / RMSE_ab(v_cpu.to(torch.float32).numpy())
    print(f"PFA_CANN_hpr_all-RMSE, PFA_CANN_hpr_Re_all-RMSE:{PFA_CANN, RMSE_CANN_Re}")
    print(f"RMSE_CANN_Components:{RMSE_CANN_ALL}")
    print(f"RMSE_CANN_Re_Components:{RMSE_CANN_Re_ALL}")
    print(f"pass_res_all:{pass_res_all}, ltPct_all:{ltPct_all}")
    print(f"pass_res:{pass_res}")
    print(f"ltPct:{ltPct}")
    exit()
