#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v1/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_quad.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v1_quad --predict-output  predict_output_v1_quad.root --regression-mode --gpus ""
#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_test.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v2_test --predict-output  predict_output_v2_test.root --regression-mode --gpus ""

#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_theta.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_theta --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""


#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v1/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix batchjob_best_epoch_state.pt --predict-output  predict_output.root --regression-mode --gpus ""

#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v2_sumPt --predict-output  predict_output_train.root --regression-mode --gpus "0"

python train.py \
--predict
--data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1.root' \
--data-config 'data/genak8_hl_features_lin_dnn.yaml' \
--network-config 'networks/mlp_genjetAK8_lin.py'  \
--model-prefix mlp_hl_lin_v1 \
--predict-output  predict_output_mlp_train.root
--regression-mode \
--gpus 0
