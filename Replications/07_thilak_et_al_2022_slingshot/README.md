# The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon

**Authors:** Vimal Thilak, Etai Littwin, Shuangfei Zhai, Omid Saremi, Roni Paiss, Joshua M. Susskind

**Paper:** [arXiv:2206.04817](https://arxiv.org/abs/2206.04817)

## Summary

This paper identifies an optimization anomaly in adaptive optimizers (Adam family) during late-stage training, dubbed the "Slingshot Mechanism." The authors observe cyclic phase transitions between stable and unstable training regimes, evidenced by cyclic behavior in the norm of the last layer's weights. Grokking occurs predominantly at the onset of these Slingshot effects.

## Dataset

- **Task:** Modular addition $(a + b) \mod p$ for various primes
- **Training fraction:** Varied (typically 50%)
- Focus on optimizer dynamics rather than specific dataset

## Model Architecture

- **Type:** Decoder-only Transformer
- **Layers:** 2
- **Model dimension:** 128
- **Attention heads:** 4
- **MLP dimension:** 512

## Training Hyperparameters

- **Optimizer:** Adam or AdamW (focus on Adam family)
- **Learning rate:** $10^{-3}$ to $10^{-4}$
- **Weight decay:** Varied, including 0 (Slingshot can occur without explicit regularization)
- **Batch size:** Full batch gradient descent
- **Training:** Extended beyond overfitting to observe Slingshot cycles

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

### Using SLURM

```bash
chmod +x run_slingshot.sh
sbatch run_slingshot.sh
```

### Running Locally (with weight decay = 0)

```bash
python train.py \
    --p=97 \
    --optimizer=adam \
    --lr=0.001 \
    --weight_decay=0.0 \
    --n_epochs=50000 \
    --log_interval=50 \
    --device=cuda
```

### With AdamW and weight decay

```bash
python train.py \
    --p=97 \
    --optimizer=adamw \
    --lr=0.001 \
    --weight_decay=1.0 \
    --n_epochs=50000 \
    --device=cuda
```

## Key Metrics to Track

The training script specifically tracks:
- **last_layer_norm**: L2 norm of output layer weights (key Slingshot indicator)
- **embedding_norm**: Norm of embedding parameters
- **transformer_norm**: Norm of transformer layer parameters
- **output_norm**: Norm of output layer (weight + bias)

## Expected Results

- **Cyclic behavior:** `last_layer_norm` should show cycles during training
- **Grokking onset:** Test accuracy jumps typically coincide with Slingshot phases
- **Works without weight decay:** Unlike many grokking phenomena, Slingshot can occur with weight_decay=0

## Analyzing Results

After training, plot the `last_layer_norm` over epochs to visualize the Slingshot cycles:

```python
import json
import matplotlib.pyplot as plt

with open('logs/training_history.json', 'r') as f:
    history = json.load(f)

plt.plot(history['epoch'], history['last_layer_norm'])
plt.xlabel('Epoch')
plt.ylabel('Last Layer Weight Norm')
plt.title('Slingshot Mechanism: Cyclic Weight Norm Behavior')
plt.show()
```

## Citation

```bibtex
@article{thilak2022slingshot,
  title={The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon},
  author={Thilak, Vimal and Littwin, Etai and Zhai, Shuangfei and Saremi, Omid and Paiss, Roni and Susskind, Joshua M},
  journal={arXiv preprint arXiv:2206.04817},
  year={2022}
}
```

