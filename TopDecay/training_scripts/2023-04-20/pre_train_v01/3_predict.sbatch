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

# 0 1 2 5 10 20 50 100 200 399 400 401 402 405 410 420 430 440 448

for epoch in  460 470 480 490 499
do

python train.py \
--predict \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--model-prefix models/pre_train_fc_pnet/v01/model_epoch-${epoch}_state.pt \
--predict-output prediction_at_epoch_${epoch} \
--weighting  \
--fetch-step 20 \
--num-workers 3 \
--fetch-by-files  \
--regression-mode \
--gpus 0 \
--network-option conv_params '[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))]' \
--network-option pnet_fc_params '[(128, 0.0)]' \
--network-option freeze_pnet 'False' \
--network-option freeze_global_fc 'False' \
--network-option globals_fc_params '[(256, 0.0), (256, 0.0)]' \
--network-option joined_fc_params '[(256, 0.0), (128, 0.0)]'

done