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
--model-prefix models/test_old_global/old/model \
--network-config TopDecay/networks/ParticleNet/conv_4_16_32_fc_01_64.py \
--tensorboard runs/test_old_global/old \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[1-2]?.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_8?.root \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
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
--batch-size 100