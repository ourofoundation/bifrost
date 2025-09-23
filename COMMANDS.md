### Dataset Creation

```bash
python bifrost/data/mp/generate_mp_dataset.py --api_key FsSFsoEfqi6J37fuRe8McMTiLyOWoVrS --max_structures 1000000 --output mp_dataset_large.json
```

```bash
python bifrost/data/mp/generate_mp_dataset.py --api_key FsSFsoEfqi6J37fuRe8McMTiLyOWoVrS --max_structures 1000 --output mp_dataset.json
```

### Training

```bash
bifrost-train --model-size small --epochs 10 --batch-size 64 --dataset bifrost/data/mp/mp_dataset.json --tensorboard
```

```bash
bifrost-train --model-size base --epochs 10 --batch-size 64 --dataset bifrost/data/mp/mp_dataset.json --tensorboard
```

```bash
# On Modal with A10G with 24GB VRAM
!cd /mnt/bifrost-data/bifrost/ && bifrost-train --model-size base --epochs 100 --save-interval 10 --batch-size 512 --warmup-steps 2000 --dataset bifrost/data/mp/mp_dataset_large.json --tensorboard

### Generation

```bash
bifrost-generate --properties '{"band_gap": 2.0}' --num-samples 1 --print-sequences --print-decoded --model-path checkpoints/checkpoint_epoch_5_small.pt
```

### Generation with property ranges

```bash
bifrost-generate --ranges '{"band_gap": [1.0, 3.0]}' --num-samples 1 --print-sequences --print-decoded --model-path checkpoints/checkpoint_epoch_10_small.pt
```