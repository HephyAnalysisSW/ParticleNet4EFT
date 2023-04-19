#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --partition g
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-gpu 40G
#SBATCH --qos short
#SBATCH --time 08:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



python train.py \
--model-prefix models/test_pnet_update_17_04/medium_particles/model \
--data-config TopDecay/data/eflow_particles_only_ctWRe_weighted.yaml \
--start-lr 1e-4 \
--num-epochs 100 \
--batch-size 100 \
--tensorboard runs/test_pnet_update_17_04/medium_particles \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[1-7]?.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--regression-mode  \
--optimizer ranger \
--lr-scheduler none \
--weighting  \
--gpus 0 \
--fetch-by-files  \
--fetch-step 10 \
--num-workers 3 \
--network-option conv_params '[(8, (32, 32, 32)), (8, (64, 64, 64)), (8, (128, 128, 128))]' \
--network-option pnet_fc_params '[(128, 0.1), (128, 0.1)]' \
--network-option freeze_pnet 'False' \
--network-option globals_fc_params '[]' \
--network-option joined_fc_params '[(64, 0.1), (64, 0.1)]'