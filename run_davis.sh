# ultrafast-test --exp-id DAVIS --config configs/saprot_agg_config.yaml --batch-size 64 --checkpoint best_models/DAVIS/davis-v7.ckpt
# rm -rf data/DAVIS/*Morgan*
# rm -rf data/DAVIS/*SaProt8
# ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --batch-size 64 --prot-proj agg --num-layers-target 2 --out-type mean
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --batch-size 64 --prot-proj agg --num-layers-target 2 --out-type mean
