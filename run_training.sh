
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_quad.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_quad --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_lin.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_lin --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""
#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_theta.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_theta --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""

python train.py \
--data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1.root' \
--data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2.root' \
--data-config 'data/genak8_hl_features_lin.yaml' \
--network-config 'networks/mlp_genjetAK8_lin.py'  \
--batch-size 256  \
--start-lr 5e-3 \
--num-epochs 2 \
--optimizer ranger \
--lr-scheduler none \
--log logs/train.log \
--regression-mode \
--gpus 0

#--model-prefix mlp_hl_lin_v1 \

#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8_small.py  --model-prefix v2_sumPt --batch-size 256  --start-lr 5e-4 --num-epochs 20 --optimizer ranger --log logs/train.log --regression-mode --gpus "0"
