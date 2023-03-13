#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --qos short
#SBATCH --time 08:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/




python train.py \
--data-train \
'/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_7*.root' \
--data-test \
'/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_8*.root' \
--data-config 'TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml' \
--network-config 'TopDecay/networks/ParticleNet/test_pnet.py' \
--model-prefix 'models/pre_train_global_fc/model' \
--batch-size 100 \
--lr-scheduler none \
--start-lr 5e-3 \
--num-epochs 400 \
--optimizer ranger \
--regression-mode \
--gpus '' \
--weighting \
--fetch-by-files \
--fetch-step 10 \
--network-option 'freeze_pnet_fc' 'True' \
--tensorboard 'runs/test_pre_train/pre_train_global_fc'