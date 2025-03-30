rm -rf data/BIOSNAP/full_data/*Morgan*
ultrafast-train --exp-id biosnap --config configs/saprot_agg_config_biosnap.yaml --batch-size 64 --InfoNCEWeight 0.05

#ultrafast-test --exp-id biosnap --config configs/saprot_agg_config_biosnap.yaml --batch-size 64 --InfoNCEWeight 0.05 --checkpoint best_models/biosnap/biosnap.ckpt