# ultrafast-test --exp-id DAVIS --config configs/saprot_agg_config.yaml --batch-size 64 --checkpoint best_models/DAVIS/davis-v7.ckpt
rm -rf data/DAVIS/*Morgan*
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --batch-size 64 