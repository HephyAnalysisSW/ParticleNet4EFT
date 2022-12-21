#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v1/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_quad.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v1_quad --predict-output  predict_output_v1_quad.root --regression-mode --gpus ""
#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_test.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v2_test --predict-output  predict_output_v2_test.root --regression-mode --gpus ""

#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_theta.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_theta --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""


#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v1/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix batchjob_best_epoch_state.pt --predict-output  predict_output.root --regression-mode --gpus ""

#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v2_sumPt --predict-output  predict_output_train.root --regression-mode --gpus "0"


# python train.py \
# --predict \
# --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2.root' \
# --data-config data/genak8_points_pf_full_theta.yaml \
# --network-config networks/particle_net_genjetAK8.py \
# --model-prefix models/pnet_test/pnet_test_epoch-0_state.pt \
# --predict-output predict_output_pnet_test.root \
# --regression-mode \
# --gpus 0 \
# --export-onnx models/pnet_test/model.onnx


PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v3/tschRefPointNoWidthRW/'


# PATH_TO_DATA='/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/'
# python train.py \
# --predict \
# --data-test \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
# --data-config 'data/genak8_points_pf_full_lin.yaml' \
# --network-config 'networks/particle_net_genjetAK8.py' \
# --model-prefix models/pnet_lin_v1/20221123-194141_particle_net_genjetAK8_ranger_lr0.005_batch256_best_epoch_state.pt \
# --predict-output predict_output_pnet_lin_v1.root \
# --regression-mode \
# --gpus 0



for epoch in {1..399..20}
do

python train.py \
--predict \
--data-test ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
--data-config 'data/genak8_hl_features_lin.yaml' \
--network-config 'networks/mlp_genjetAK8_lin.py'  \
--model-prefix models/mlp_hl_lin_test_4/20221221-104334_mlp_genjetAK8_lin_ranger_lr0.0005_batch1000_epoch-${epoch}_state.pt \
--predict-output prediction_at_epoch_${epoch}.root \
--regression-mode \
--gpus 0 

done


# \
# --export-onnx models/mlp_hl_lin_test1/model.onnx
