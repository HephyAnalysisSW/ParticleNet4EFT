
python train.py --predict --data-test '/groups/hephy/cms/robert.schoefbeck/TMB/postprocessed/gen/v1/tschRefPointNoWidthRW/tschRefPointNoWidthRW_2*.root' --data-config data/genak8_points_pf_full.yaml --network-config networks/particle_net_genjetAK8.py --model-prefix batchjob_best_epoch_state.pt --predict-output  predict_output.root --regression-mode --gpus ""
