#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --partition g
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-gpu 20G
#SBATCH --qos short
#SBATCH --time 03:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



python train.py \
--model-prefix models/test_target_globals/model \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--tensorboard runs/test_target_globals \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[1-2]?.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_8?.root \
--data-config TopDecay/data/eflow_particles_target_globals.yaml \
--regression-mode  \
--optimizer ranger \
--start-lr 1e-4 \
--lr-scheduler none \
--weighting  \
--gpus 0 \
--fetch-by-files  \
--fetch-step 10 \
--num-workers 3 \
--num-epochs 300 \
--batch-size 100 \
--network-option conv_params '[(4, (8, 8, 8)), (4, (16, 16, 16))]' \
--network-option pnet_fc_params '[(64, 0.1)]' \
--network-option globals_fc_params '[(64, 0.1), (64, 0.1), (64, 0.1)]' \
--network-option joined_fc_params '[(64,0.1)]'