#rm -rf data/BIOSNAP/full_data/*Morgan*
ultrafast-train --exp-id biosnap --config configs/saprot_agg_config_biosnap.yaml --batch-size 128 --InfoNCEWeight 0.25
#ultrafast-train --exp-id biosnap --config configs/saprot_agg_config_biosnap.yaml --batch-size 128 --InfoNCEWeight 0.05 --checkpoint best_models/biosnap/biosnap-v5_ep161.ckpt --epochs 90
