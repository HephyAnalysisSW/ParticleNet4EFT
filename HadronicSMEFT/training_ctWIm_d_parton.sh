#    --data-test  '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/*.root'\
#    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_865.root'\

ipython -i train.py -- \
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_[0-9][0-9].root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctWIm_d_parton.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobal_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctWIm-v2/ctWIm_d_parton\
    --batch-size 128  --start-lr 5e-4 --num-epochs 20 --optimizer ranger\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --tensorboard test\
    --gpus 0\
    --load-epoch -1
