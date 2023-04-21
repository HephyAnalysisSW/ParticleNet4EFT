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
--model-prefix models/test_pre_train_globals/model \
--num-epochs 10 \
--tensorboard runs/test_pre_train_globals/pre_train_globals \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_0.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1.root \
--regression-mode  \
--optimizer ranger \
--start-lr 1e-4 \
--lr-scheduler none \
--weighting  \
--gpus 0 \
--fetch-by-files  \
--fetch-step 10 \
--num-workers 3 \
--batch-size 100 \
--network-option conv_params '[(4, (8,8,8)), (8, (16, 16, 16))]' \
--network-option pnet_fc_params '[(16, 0.1), (16, 0.1)]' \
--network-option freeze_pnet 'True' \
--network-option freeze_global_fc 'False' \
--network-option globals_fc_params '[(10, 0.1), (10, 0.1)]' \
--network-option joined_fc_params '[(5, 0.1)]'