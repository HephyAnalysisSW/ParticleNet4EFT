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
--model-prefix models/test_ParticleNetEFT/model \
--network-config TopDecay/networks/ParticleNet/PNGLobal_TopDecay_config.py \
--tensorboard runs/test_ParticleNetEFT \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_0.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1.root \
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
--num-epochs 100 \
--batch-size 100 \
--network-option conv_params '[(2, (4, 4, 4)), (2, (8, 8, 8))]' \
--network-option pnet_fc_params '[(16, 0.1)]' \
--network-option globals_fc_params '[(64, 0.1)]' \
--network-option joined_fc_params '[(64,0.1)]'