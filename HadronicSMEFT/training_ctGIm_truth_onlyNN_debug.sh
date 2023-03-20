
#    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root'\
#    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root'\
ipython -i train.py --\
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm_truth_onlyNN.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetNN_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGIm-truth-onlyNN-debug-noDO/ctGIm-truth\
    --start-lr 1e-2 --num-epochs 50 --optimizer adam --tensorboard "./pNet"\
    --batch-size 10000\
    --steps-per-epoch 100\
    --in-memory\
    --log HadronicSMEFT/logs/train.log\
    --fetch-by-files\
    --fetch-step 10\
    --regression-mode\
    --weighting\
    --gpus 0
    #--load-epoch -1
    #--lr-finder "1e-4,5e-3,10"\
