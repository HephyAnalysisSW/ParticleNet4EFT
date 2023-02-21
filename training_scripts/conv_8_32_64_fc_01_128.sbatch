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

#### mlp global features full weighted 
DATA_CONFIG='data/eflow_particles_only_ctWRe_weighted.yaml'
NETWORK_CONFIG='networks/ParticleNet/conv_8_32_64_fc_01_128.py'
MODEL_PREFIX=${NETWORK_CONFIG#networks/ParticleNet/}
MODEL_PREFIX=${MODEL_PREFIX%.py}_${DATA_CONFIG#data/}
MODEL_PREFIX=models/${MODEL_PREFIX%.yaml}/model
BATCH_SIZE=200
START_LR=1e-4
NUM_EPOCHS=100


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
--tensorboard ${MODEL_PREFIX}



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