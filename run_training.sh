
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_quad.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_quad --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_lin.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v3_lin --batch-size 256  --start-lr 5e-3 --num-epochs 30 --optimizer ranger --log logs/train.log --regression-mode --gpus 1
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_theta.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_theta --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""

#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_sumPt --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_lin.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_lin --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""



###### train full particle net 
# PATH_TO_DATA='/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/'
# python train.py \
# --data-train \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[1-7]?.root' \
# --data-test \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
# --data-config data/genak8_points_pf_full_lin.yaml \
# --network-config networks/particle_net_genjetAK8.py \
# --model-prefix models/pnet_lin_v1/20221123-194141_particle_net_genjetAK8_ranger_lr0.005_batch256 \
# --batch-size 256  \
# --num-workers 3 \
# --start-lr 5e-3 \
# --num-epochs 20 \
# --optimizer ranger \
# --lr-scheduler none \
# --regression-mode \
# --load-epoch 9 \
# --gpus 0


###### train DNN on hl_features, if not run with 'in-memory' training seems to pause every few seconds...
#PATH_TO_DATA='/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/'
# PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v3/tschRefPointNoWidthRW/'
# python train.py \
# --data-train \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[1-7]?.root' \
# --data-test \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
# --data-config 'data/genak8_hl_features_lin.yaml' \
# --network-config 'networks/mlp_genjetAK8_lin.py'  \
# --batch-size 1000  \
# --num-workers 3 \
# --start-lr 5e-4 \
# --num-epochs 400 \
# --optimizer ranger \.
# --lr-scheduler none \
# --regression-mode \
# --gpus 0 \
# --model-prefix models/mlp_hl_lin_test_4/{auto} \
# --in-memory \
# --steps-per-epoch 1000

# ${PATH_TO_DATA}'tschRefPointNoWidthRW_2*.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_3*.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_4*.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_5*.root' \

# === gen particles ===
# PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v3/tschRefPointNoWidthRW/'
# DATA_CONFIG='data/genak8_hl_features_lin.yaml'

# === delphes detector sim ===
PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v5/tschRefPointNoWidthRW/'
DATA_CONFIG='data/delphes_hl_features_lin.yaml'
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[1-7]?.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \

# python train.py \
# --data-train \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
# --data-test \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
# --data-config ${DATA_CONFIG} \
# --network-config 'networks/mlp_genjetAK8_lin.py' \
# --batch-size 1000 \
# --num-workers 3 \
# --start-lr 5e-4 \
# --num-epochs 1 \
# --optimizer ranger \
# --lr-scheduler none \
# --regression-mode \
# --gpus 0 \
# --model-prefix models/mlp_hl_lin_delphes_test_tensorboard/mlp \
# --fetch-by-files \
# --fetch-step 10 \
# --tensorboard tensorboard \


# --in-memory \
# --steps-per-epoch 1

# === weighted particle net ===

python train.py \
--data-train \
'/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_70.root' \
--data-test \
'/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_71.root' \
--data-config 'data/ak8_points_pf_full_weighttest.yaml' \
--network-config 'networks/particle_net_genjetAK8_weighttest.py' \
--model-prefix 'models/ParticleNet_weighted_test_visualization/pNet' \
--batch-size 100 \
--lr-scheduler none \
--start-lr 5e-3 \
--num-epochs 1 \
--optimizer ranger \
--regression-mode \
--gpus 0 \
--weighting \
--fetch-by-files \
--fetch-step 10 \
--tensorboard pNet_visualization \