#!/bin/bash

# Run ablations

ablations=(baseline ablation1 ablation2)

for ablation in "${ablations[@]}"; do
#    CUDA_VISIBLE_DEVICES=0,1,2 bash ./scripts/dist_train.sh 3 13232 \
#	    --config ./cfgs/NRG_models/AdaPoinTr_${ablation}.yaml \
#	    --exp_name NRG_${ablation}_exp

    mkdir -p results/NRG_${ablation}
    python utils/run_tests.py --cfg_path experiments/AdaPoinTr_${ablation}/NRG_models/NRG_${ablation}_exp/config.yaml \
                              --ckpt_path experiments/AdaPoinTr_${ablation}/NRG_models/NRG_${ablation}_exp/ckpt-best.pth \
                              --test_txt_path data/NRG_${ablation}/test.txt --out_path results/NRG_${ablation}

done

echo "All ablations done"
