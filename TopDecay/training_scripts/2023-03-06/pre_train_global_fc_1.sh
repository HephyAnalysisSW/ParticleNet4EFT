#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --qos short
#SBATCH --time 08:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



python train.py \
--model-prefix models/pre_train_global_fc_1/model \
--num-epochs 400 \
--batch-size 100 \
--tensorboard runs/test_pre_train/pre_train_global_fc_1 \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[4-7]*.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_8*.root \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--regression-mode  \
--start-lr 1e-4 \
--optimizer ranger \
--lr-scheduler none \
--weighting  \
--gpus 0 \
--fetch-by-files  \
--fetch-step 20 \
--num-workers 3 \
--network-option conv_params '[(4, (8,8,8)), (8, (16, 16, 16))]' \
--network-option pnet_fc_params '[(32, 0.1)]' \
--network-option freeze_pnet 'True' \
--network-option globals_fc_params '[(300, 0.1), (200, 0.1), (100, 0.1)]' \
--network-option joined_fc_params '[(50, 0.1)]'