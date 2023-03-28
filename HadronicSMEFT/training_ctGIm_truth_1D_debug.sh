#    --data-test  '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/*.root'\

ipython -i train.py --\
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9].root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm_truth.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobalSmall1D_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGIm-truth-noBNS-redCN-debug/ctGIm-truth\
    --batch-size 128  --start-lr 1e-4 --num-epochs 20 --optimizer ranger --tensorboard "./pNet"\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --gpus ""
#    --load-epoch -1
