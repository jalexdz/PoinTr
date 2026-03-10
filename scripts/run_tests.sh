#!/bin/bash

# Run ablations
python utils/run_tests.py --out_path ./results \
                          --ablation "Baseline" experiments/AdaPoinTr_baseline/NRG_models/NRG_baseline_exp/config.yaml experiments/AdaPoinTr_baseline/NRG_models/NRG_baseline_exp/ckpt-best.pth data/NRG_baseline/test.txt --out_path results/NRG_baseline \
                          --ablation "Ablation 1" experiments/AdaPoinTr_ablation1/NRG_models/NRG_ablation1_exp/config.yaml experiments/AdaPoinTr_ablation1/NRG_models/NRG_ablation1_exp/ckpt-best.pth data/NRG_ablation1/test.txt --out_path results/NRG_ablation1 \
                          --ablation "Ablation 2" experiments/AdaPoinTr_ablation2/NRG_models/NRG_ablation2_exp/config.yaml experiments/AdaPoinTr_ablation2/NRG_models/NRG_ablation2_exp/ckpt-best.pth data/NRG_ablation2/test.txt --out_path results/NRG_ablation2

# ablations=("Baseline" ablation1 ablation2)

# for ablation in "${ablations[@]}"; do
# #    CUDA_VISIBLE_DEVICES=0,1,2 bash ./scripts/dist_train.sh 3 13232 \
# #	    --config ./cfgs/NRG_models/AdaPoinTr_${ablation}.yaml \
# #	    --exp_name NRG_${ablation}_exp

#     mkdir -p results/NRG_${ablation}
#     python utils/run_tests.py --cfg_path experiments/AdaPoinTr_${ablation}/NRG_models/NRG_${ablation}_exp/config.yaml \
#                               --ckpt_path experiments/AdaPoinTr_${ablation}/NRG_models/NRG_${ablation}_exp/ckpt-best.pth \
#                               --test_txt_path data/NRG_${ablation}/test.txt --out_path results/NRG_${ablation}

# done

echo "All ablations done"
