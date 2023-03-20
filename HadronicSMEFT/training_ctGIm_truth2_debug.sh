#    --in-memory\
ipython -i train.py --\
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm_truth2.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobalSmall_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGIm-truth2-noBNS-redCN-debug/ctGIm-truth\
    --start-lr 1e-2 --num-epochs 50 --optimizer adam --tensorboard "./pNet"\
    --batch-size 10000\
    --steps-per-epoch 100\
    --log HadronicSMEFT/logs/train.log\
    --fetch-by-files\
    --fetch-step 10\
    --regression-mode\
    --weighting\
    --in-memory\
    --gpus 0

