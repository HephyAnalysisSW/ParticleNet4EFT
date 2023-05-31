#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --partition g
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-gpu 40G
#SBATCH --qos medium
#SBATCH --time 1-00:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



python train.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--model-prefix models/pnet_comparison/globals/model \
--tensorboard runs/pnet_comparison/globals \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_0?.root /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[1-7]?.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
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
--num-epochs 100 \
--network-option conv_params '[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))]' \
--network-option pnet_fc_params '[(128, 0.0)]' \
--network-option globals_fc_params '[(256, 0.0), (256, 0.0)]' \
--network-option joined_fc_params '[(256, 0.0), (128, 0.0)]'