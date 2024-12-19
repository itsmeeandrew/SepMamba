# SepMamba: State-space models for speaker separation using Mamba

<p align=center><em>
Thor Højhus Avenstrup, Boldizsár Elek, István László Mádi, András Bence Schin,<br />
Morten Mørup, Bjørn Sand Jensen, Kenny Olsen <br />
Technical University of Denmark (DTU)  <br />
<a href="https://arxiv.org/abs/2410.20997">Paper on arXiv</a>
</em></p>


**Deep learning-based single-channel speaker separa- tion has improved significantly in recent years in large part due to the introduction of the transformer-based attention mechanism. However, these improvements come with intense computational demands, precluding their use in many practical applications. As a computationally efficient alternative with similar modeling capabilities, Mamba was recently introduced. We propose Sep- Mamba, a U-Net-based architecture composed of bidirectional Mamba layers. We find that our approach outperforms similarly- sized prominent models — including transformer-based models — on the WSJ0 2-speaker dataset while enjoying significant computational benefits in terms of multiply-accumulates, peak memory usage, and wall-clock time. We additionally report strong results for causal variants of SepMamba. Our approach provides a computationally favorable alternative to transformer- based architectures for deep speech separation.**

![network](https://github.com/user-attachments/assets/3f8897ee-0297-4464-901c-12befc0a1a46)

| Model                                      | SI-SNRi | SI-SDRi | SDRi  | # Params | GMAC/s | Fw. pass (ms) | Mem. Usage (GB) |
|--------------------------------------------|---------|---------|-------|----------|--------|---------------|-----------------|
| Conv-TasNet                                | 15.3    | -       | 15.6  | 5.1M     | 2.82   | 30.79         | 1.13            |
| DualPathRNN                                | 18.8    | -       | 19.0  | 2.6M     | 42.52  | 101.83        | 7.31            |
| SudoRM-RF                                  | 18.9    | -       | -     | 2.6M     | 2.58   | 69.23         | 1.60            |
| **SepFormer**                              | 20.4    | -       | -     | 26M      | 257.94 | 189.25        | 35.30           |
| **SepFormer + DM**                         | 22.3    | -       | -     | 26M      | 257.94 | 189.25        | 35.30           |
| MossFormer (S)                             | -       | 20.9    | -     | 10.8M    | -      | -             | -               |
| MossFormer (M) + DM                        | -       | 22.5    | -     | 25.3M    | -      | -             | -               |
| MossFormer (L) + DM                        | -       | 22.8    | -     | 42.1M    | 70.4   | 72.71         | 9.57            |
| MossFormer2 + DM                           | -       | 24.1    | -     | 55.7M    | 84.2   | 97.60         | 12.30           |
| TF-GridNet (S)                             | -       | 20.6    | -     | 8.2M     | 19.2   | -             | -               |
| TF-GridNet (M)                             | -       | 22.2    | -     | 8.4M     | 36.2   | -             | -               |
| TF-GridNet (L)                             | -       | 23.4    | 23.5  | 14.4M    | 231.1  | -             | -               |
| SP-Mamba                                   | 22.5    | -       | -     | 6.14M    | 119.35 | 148.11        | 14.40           |
| **SepMamba (S) + DM (ours)**               | **21.2**    | **21.2**    | **21.4**  | **7.2M**     | **12.46**  | **17.84**         | **2.00**            |
| **SepMamba (M) + DM (ours)**               | **22.7**    | **22.7**    | **22.9**  | **22M**      | **37.0**   | **27.25**         | **3.04**            |


## Environment

Supported operating systems: Linux (Nvidia GPUs from volta gen. or later)

To install the necessary dependencies, simply run:

```bash
make requirements
```

## Logging

For logging we use `wandb` therefore `wandb` needs to be set up in the directory by using:

```
wandb login
```

The `wandb` project's name can be set in `src/conf/config.yaml`:

```yaml
wandb: 
  project_name: sepmamba-speaker-separation
```

To change the output directory, navigate to `src/conf/config.yaml` and change the line:

```yaml
hydra:
  run:
    dir: path/to/output/dir/${experiment_name}
```

## Data

Before starting training, the data paths need to be set up. The training dataset is specified in `src/conf/dataset/2mix_dyn.yaml`:

```yaml
root_dir: /path/to/wsj0
```

where `root_dir` should point to the raw `wsj0` dataset.

The validation dataset path is specified in `src/conf/config.yaml`:

```yaml
val_dataset:
  root_dir: /path/to/wsj0-mix/2speakers/wav8k/min/
```

where `root_dir` is the validation dataset path. We use the test set mixtures created by [pywsj0-mix](https://github.com/mpariente/pywsj0-mix) scripts.

## Training

To start a training run with the noncausal model:

```python
python src/train.py model="SepMamba" wandb.experiment_name="SepMamba_experiment"
```
To start a training run with the causal model:

```python
python src/train.py model="SepMambaCausal" wandb.experiment_name="SepMamba_experiment"
```

`experiment_name` will be the name of the output folder and the wandb run name.

The default settings contain the parameters for the smaller models. The hyperparameters can be changed in `src/conf/models`.

The medium size noncausal config:

```yaml
dim: 128
kernel_sizes:
  - 16
  - 16
  - 16
strides:
  - 2
  - 2
  - 2
act_fn: "relu"
num_blocks: 3
```

The medium size causal config:

```yaml
dim: 128
kernel_sizes:
  - 16
  - 16
  - 16
strides:
  - 2
  - 2
  - 2
act_fn: "relu"
num_blocks: 6
```

## Continue a run

To continue a started run from a checkpoint:

```python
python src/train.py \
    wandb.experiment_name="SepMamba_experiment" \
    load.load_checkpoint=True
    load.checkpoint_path="path/to/checkpoint" \
    lr_scheduler.gamma=1.0 \
```
where:
- `checkpoint_path`: Path to the checkpoint.
- `lr_scheduler.gamma`: Sets the gamma value of the StepLR scheduler.
