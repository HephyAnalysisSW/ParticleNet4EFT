#    --data-test  '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/*.root'\

python train.py \
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm_truth.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobalSmall_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGIm-truth-noBNS/ctGIm-truth\
    --batch-size 128  --start-lr 5e-4 --num-epochs 10 --optimizer ranger\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --tensorboard test\
    --gpus 0\
    --load-epoch -1
