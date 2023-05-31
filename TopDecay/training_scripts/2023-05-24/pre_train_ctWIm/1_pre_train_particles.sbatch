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
--model-prefix models/pre_train_fc_pnet/ctWIm_v02/model \
--num-epochs 430 \
--load-epoch 399 \
--tensorboard runs/pre_train_fc_pnet/ctWIm_v02/pre_train_particles \
--network-config TopDecay/networks/ParticleNet/PNGLobal_TopDecay_config.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWIm_weighted.yaml \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[1-6]??.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_7??.root \
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
--network-option conv_params '[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))]' \
--network-option pnet_fc_params '[(128, 0.0)]' \
--network-option freeze_pnet 'False' \
--network-option freeze_global_fc 'True' \
--network-option globals_fc_params '[(256, 0.0), (256, 0.0)]' \
--network-option joined_fc_params '[(256, 0.0), (128, 0.0)]'