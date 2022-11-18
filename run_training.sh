
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_quad.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_quad --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_lin.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_lin --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""


# python train.py \
# --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1.root' \
# --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2.root' \
# --data-config data/genak8_points_pf_full_theta.yaml \
# --network-config networks/particle_net_genjetAK8.py \
# --model-prefix models/pnet_test/pnet_test \
# --batch-size 256  \
# --start-lr 5e-3 \
# --num-epochs 1 \
# --optimizer ranger \
# --lr-scheduler none \
# --regression-mode \
# --gpus 0



PATH_TO_DATA='/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/'

python train.py \
--data-train \
${PATH_TO_DATA}'tschRefPointNoWidthRW_?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_1?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_2?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_3?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_4?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_5?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_6?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_7?.root' \
--data-test \
${PATH_TO_DATA}'tschRefPointNoWidthRW_8?.root' \
${PATH_TO_DATA}'tschRefPointNoWidthRW_9?.root' \
--data-config 'data/genak8_hl_features_lin.yaml' \
--network-config 'networks/mlp_genjetAK8_lin.py'  \
--batch-size 1000  \
--num-workers 1 \
--start-lr 5e-4 \
--num-epochs 800 \
--optimizer ranger \
--lr-scheduler none \
--log logs/train.log \
--regression-mode \
--gpus 0 \
--model-prefix models/mlp_hl_lin_v1/{auto} \
--in-memory \
--steps-per-epoch 1000


# ${PATH_TO_DATA}'tschRefPointNoWidthRW_2*.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_3*.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_4*.root' \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_5*.root' \
