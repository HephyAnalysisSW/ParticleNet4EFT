#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v9/TT01jDebug/TT01jDebug_*.root' --data-config HadronicSMEFT/data/delphesJet_ctGIm.yaml --predict-config HadronicSMEFT/predict/predict.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm/output.root #SPLIT199
#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v9/TT01jDebug/TT01jDebug_*.root' --data-config HadronicSMEFT/data/delphesJet_ctGIm.yaml --predict-config HadronicSMEFT/predict/predict-v2.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/TT01jDebug-v2/output.root #SPLIT199

#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_d_parton.yaml --predict-config HadronicSMEFT/predict/predict-d_parton.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/TT01j_HT800_ext_comb-d_parton/output.root #SPLIT99

# truth test
#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_truth.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm_truth.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm-truth/TT01j_HT800_ext_comb/output.root #SPLIT99

# truth test
#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_truth.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm_truth.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm-truth-noBNS/TT01j_HT800_ext_comb/output.root #SPLIT99
#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_truth.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm_truth.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm-truth-noBNS-redCN-debug/TT01j_HT800_ext_comb/output.root #SPLIT99

# NN only
#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_truth_NN.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm_NN_truth.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm-truth-NN/TT01j_HT800_ext_comb/output.root #SPLIT99

#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_truth_onlyNN.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm_NN_truth.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm-truth-onlyNN-v2-oldData/TT01j_HT800_ext_comb/output.root #SPLIT99 

#python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root' --data-config HadronicSMEFT/data/delphesJet_ctGIm_truth2.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm_truth2.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm_truth2/TT01j_HT800_ext_comb/output.root #SPLIT99 

python predict.py --gpus "" --data-test '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root' --data-config HadronicSMEFT/data/delphesJet_ctGIm.yaml --predict-config HadronicSMEFT/predict/predict-ctGIm.yaml --predict-output /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/predictions/ctGIm/TT01j_HT800_ext_comb/output.root #SPLIT99 
