#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_quad.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v2_quad --predict-output  predict_output_v2_quad.root --regression-mode --gpus ""
#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_test.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix v2_test --predict-output  predict_output_v2_test.root --regression-mode --gpus ""

#python train.py --data-train '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_1*.root' --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_theta.yaml --network-config networks/particle_net_genjetAK8.py  --model-prefix v2_theta --batch-size 256  --start-lr 5e-3 --num-epochs 2 --optimizer ranger --log logs/train.log --regression-mode --gpus ""


#python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full_sumPt.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix batchjob_best_epoch_state.pt --predict-output  predict_output.root --regression-mode --gpus ""

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


# === gen particles ===
# PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/TMB/postprocessed/gen/v3/tschRefPointNoWidthRW/'
# DATA_CONFIG='data/genak8_hl_features_lin.yaml'

# === delphes detector sim ===
# PATH_TO_DATA='/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v5/tschRefPointNoWidthRW/'
# DATA_CONFIG='TopDecay/data/delphes_hl_features_lin.yaml'


# PATH_TO_DATA='/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v2/tschRefPointNoWidthRW/'
# python train.py \
# --predict \
# --data-test \
# ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
# --data-config 'data/genak8_points_pf_full_lin.yaml' \
# --network-config 'networks/particle_net_genjetAK8.py' \
# --model-prefix models/pnet_lin_v2/20221123-194141_particle_net_genjetAK8_ranger_lr0.005_batch256_best_epoch_state.pt \
# --predict-output predict_output_pnet_lin_v2.root \
# --regression-mode \
# --gpus 0

pwd
cd ..
pwd

for epoch in 0 1 5 10 20 40 50 60 99 200 399
do

python train.py \
--predict \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--model-prefix models/test_pnet_update_17_04/freeze_pnet/model_epoch-${epoch}_state.pt \
--predict-output prediction_at_epoch_${epoch} \
--weighting  \
--fetch-step 20 \
--num-workers 3 \
--fetch-by-files  \
--regression-mode \
--gpus 0 \
--network-option conv_params '[(4, (8,8,8)), (8, (16, 16, 16))]' \
--network-option pnet_fc_params '[(32, 0.1)]' \
--network-option freeze_pnet 'True' \
--network-option globals_fc_params '[(300, 0.1), (200, 0.1), (100, 0.1)]' \
--network-option joined_fc_params '[(50, 0.1)]'

python train.py \
--predict \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--model-prefix models/test_pnet_update_17_04/large_globals/model_epoch-${epoch}_state.pt \
--predict-output prediction_at_epoch_${epoch} \
--weighting  \
--fetch-step 20 \
--num-workers 3 \
--fetch-by-files  \
--regression-mode \
--gpus 0 \
--network-option conv_params '[(16, (64, 64, 64)), (16, (128, 128, 128)), (16, (256, 256, 256))]' \
--network-option pnet_fc_params '[(256, 0.1), (256, 0.1)]' \
--network-option freeze_pnet 'False' \
--network-option globals_fc_params '[(200, 0.1), (200, 0.1)]' \
--network-option joined_fc_params '[(256, 0.1), (256, 0.1)]'


python train.py \
--predict \
--data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_[8-9]?.root \
--network-config TopDecay/networks/ParticleNet/test_pnet.py \
--data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
--model-prefix models/test_pnet_update_17_04/medium_globals/model_epoch-${epoch}_state.pt \
--predict-output prediction_at_epoch_${epoch} \
--weighting  \
--fetch-step 20 \
--num-workers 3 \
--fetch-by-files  \
--regression-mode \
--gpus 0 \
--network-option conv_params '[(8, (32, 32, 32)), (8, (64, 64, 64)), (8, (128, 128, 128))]' \
--network-option pnet_fc_params '[(128, 0.1), (128, 0.1)]' \
--network-option freeze_pnet 'False' \
--network-option globals_fc_params '[(200, 0.1), (200, 0.1)]' \
--network-option joined_fc_params '[(128, 0.1), (128, 0.1)]'

done



# python train.py \
# --predict \
# --data-test ${PATH_TO_DATA}'tschRefPointNoWidthRW_[8-9]?.root' \
# --data-config 'data/genak8_hl_features_lin.yaml' \
# --network-config 'networks/mlp_genjetAK8_lin.py'  \
# --model-prefix models/mlp_hl_lin_test_5/20230109-153813_mlp_genjetAK8_lin_ranger_lr0.0005_batch1000_epoch-19_state.pt \
# --predict-output prediction_at_epoch_19.root \
# --regression-mode \
# --gpus 0 


# python predict.py \
# --gpus "" \
# --data-test /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v6/tschRefPointNoWidthRW/tschRefPointNoWidthRW_8?.root \
# --data-config TopDecay/data/eflow_particles_delphes_globals_ctWRe_weighted.yaml \
# --predict-config TopDecay/predict/predict_ctWRe.yaml \
# --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm/TT01j_HT800_ext_comb/output.root 
