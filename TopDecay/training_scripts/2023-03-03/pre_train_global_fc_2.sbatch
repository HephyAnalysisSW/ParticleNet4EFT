#!/bin/bash
#

#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --qos short
#SBATCH --time 08:00:00
#SBATCH -D /users/oskar.rothbacher/CMS/ParticleNet4EFT/



train.py \
--model-prefix models/pre_train_global_fc_2/model \
--batch-size 200 \
--data-train /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_7*.root \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_8*.root \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--regression-mode  \
