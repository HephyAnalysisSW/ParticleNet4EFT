#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --partition g
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu 40G
#SBATCH --cpus-per-task 4
#SBATCH --qos short
#SBATCH --time 08:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/'

#### particle net with global features large weighted 
DATA_CONFIG='data/ak8_points_pf_only_full.yaml'
NETWORK_CONFIG='networks/particle_net_genjetAK8_weighttest.py'
MODEL_PREFIX='models/pnet-hl-features_weighted_full_no_gf_v1/pnet'
BATCH_SIZE=100
START_LR=1e-3
NUM_EPOCHS=30


#### training
python train.py \
--data-train \
${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_[1-7]?.root' \
--data-test \
${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
--data-config ${DATA_CONFIG} \
--network-config ${NETWORK_CONFIG} \
--batch-size ${BATCH_SIZE} \
--num-workers 3 \
--start-lr ${START_LR} \
--num-epochs ${NUM_EPOCHS} \
--optimizer ranger \
--lr-scheduler none \
--regression-mode \
--weighting \
--gpus 0 \
--model-prefix ${MODEL_PREFIX} \
--fetch-by-files \
--fetch-step 10 \
--tensorboard pnet_only_large



#### LR finder
# python train.py \
# --data-train \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[1-7]?.root' \
# --data-test \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
# --data-config ${DATA_CONFIG} \
# --network-config ${NETWORK_CONFIG} \
# --batch-size ${BATCH_SIZE} \
# --num-workers 3 \
# --optimizer ranger \
# --regression-mode \
# --weighting \
# --gpus 0 \
# --fetch-by-files \
# --fetch-step 10 \
# --lr-finder "5e-7, 5e0, 100"