#    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root'\

ipython -i train.py --\
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_0.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm_truth_NN.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobalNN_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGIm-truth-NN-debug-v2/ctGIm-truth\
    --batch-size 128  --start-lr 1e-3 --num-epochs 20 --optimizer adam --tensorboard "./pNet"\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --gpus ""
    #--load-epoch -1
