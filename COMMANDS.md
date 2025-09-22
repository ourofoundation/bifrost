cd /Users/mmoderwell/ouro/bifrost && bifrost-generate --properties '{"band_gap": 2.0, "density": 4.0}' --num-samples 1 --print-sequences --print-decoded --model-path checkpoints/checkpoint_epoch_10_small.pt


cd /Users/mmoderwell/ouro/bifrost && bifrost-generate --properties '{"band_gap": 2.0}' --num-samples 1 --print-sequences --print-decoded --model-path checkpoints/checkpoint_epoch_5_small.pt


cd /Users/mmoderwell/ouro/bifrost && python bifrost/data/mp/generate_mp_dataset.py --api_key FsSFsoEfqi6J37fuRe8McMTiLyOWoVrS --max_structures 100 --output mp_dataset.json

