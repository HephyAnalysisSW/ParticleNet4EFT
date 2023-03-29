#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --partition g
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-gpu 40G
#SBATCH --qos short
#SBATCH --time 08:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



python train.py \
--model-prefix models/test_lr_5/model \
--start-lr 1e-5 \
--tensorboard runs/test_lr/test_lr_5 \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[1-7]?.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--regression-mode  \
--optimizer ranger \
--lr-scheduler none \
--weighting  \
--gpus 0 \
--fetch-by-files  \
--fetch-step 10 \
--num-workers 3 \
--num-epochs 200 \
--batch-size 100 \
--network-option conv_params '[(8, (32, 32, 32)), (8, (64, 64, 64))]' \
--network-option pnet_fc_params '[(128, 0.1)]' \
--network-option globals_fc_params '[(200, 0.1), (100, 0.1)]' \
--network-option joined_fc_params '[(50, 0.1)]'