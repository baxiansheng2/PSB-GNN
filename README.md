# DEDP-GNN
The implementation of the paper "**Robust Graph Learning via Distilling Explainability and Dual-Channel Priors**". DEDP-GNN is a defense framework that integrates Explainability-Guided Structural Edge Noise Injection (SENI) and Feature-Structure Fusion to achieve robust graph learning.

## Main Structure

- **models**: implementation of GNN models (DEDP-GNN and baselines)
- **victims**: experiments for training
  - `train.py`: script for training DEDP-GNN (Ours)
  - `trainbaseline.py`: script for training baselines (GCN, RGCN, etc.) and Teacher models
  - `generate_gnnexplainer.py`: script for generating structural priors
- **attackers**: implementation of attack methods
- **attack**: experiments for attacking
  - `gen_attack.py`: generate adversarial perturbations
  - `evasion_attack.py`: evaluate model robustness under evasion attack
  - `perturbed_adjs`: saved adversarial adjacency matrices
- **common**: utility functions
- **res**: saved GNNExplainer priors (.npy files)
- **victims/models**: saved model checkpoints (including pre-trained teachers)

## Requirements
- python >= 3.8
- pytorch >= 1.10
- torch_geometric
- deeprobust
- torch_sparse
- torch_scatter
- pyyaml

## Running Step

### 0. Preparation (Generate Structural Priors)
Before training DEDP-GNN, generate the edge importance mask using GNNExplainer.
```
> cd victims
> python generate_gnnexplainer.py --dataset cora --teacher_path ./models/gcn-cora-teacher.pth --num_hidden 16 --sample_ratio 1.0 --device_id 0

```

### 1. Training Models
**Option A: Training Teacher Model (Optional)**
```
> cd victims
> python trainbaseline.py --model gcn --dataset cora --num_hidden 16 --save_name gcn-cora-teacher --gpu_id 0
```

**Option B: Training DEDP-GNN (Ours)**
```
>cd victims
>python trainbaseline.py --model gcn --dataset cora --num_hidden 16 --save_name gcn-cora-teacher --gpu_id 0  (train teacher model)

>python train.py --dataset cora --model EGNDgcn --num_hidden 16 --teacher_path ./victims/models/gcn-cora-teacher.pth --pni_path ./res/Cora/gnnexplainer_init_full.npy --pni_reg 1.0 --distill_lambda 5.0 --lambda_base 0.7 --lambda_range 0.3 --bias 0.6 --threshold 0.0 --lr 0.005 --epochs 600 --device_id 0
```

**Option C: Training Baselines (e.g., RGCN)**
```
>cd victims
> python trainbaseline.py --dataset cora --model rgcn --num_hidden 16 --device_id 0
```


### 2. Performing Attacks (Generate Adversarial Samples)
Generate perturbed adjacency matrix using attack methods (e.g., PGA) targeting the trained model.
```
> cd attack
> python gen_attack.py --dataset cora --attack pga --victim egnd --ptb_rate 0.05 --gpu_id 0 --save True
```

### 3. Evaluation (Evasion Attack)
Evaluate the robust accuracy using the generated adversarial graph. For DEDP-GNN, specific inference parameters (bias/threshold) are applied here.

**Evaluate DEDP-GNN (with Inference Pruning):**
```
> python evasion_attack.py --dataset cora --victim egnd --attack pga --ptb_rate 0.05 --bias 0.35 --threshold 0.15 --gpu_id 0
```

**Evaluate Baselines (e.g., RGCN):**
```
> python evasion_attack.py --dataset cora --victim rgcn --attack pga --ptb_rate 0.05 --gpu_id 0
```
