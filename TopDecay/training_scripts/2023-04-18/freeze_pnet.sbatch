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
--model-prefix models/test_pnet_update_17_04/freeze_pnet/model \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--start-lr 1e-4 \
--num-epochs 400 \
--batch-size 100 \
--tensorboard runs/test_pnet_update_17_04/freeze_pnet \
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
--network-option conv_params '[(4, (8,8,8)), (8, (16, 16, 16))]' \
--network-option pnet_fc_params '[(32, 0.1)]' \
--network-option freeze_pnet 'True' \
--network-option globals_fc_params '[(300, 0.1), (200, 0.1), (100, 0.1)]' \
--network-option joined_fc_params '[(50, 0.1)]'