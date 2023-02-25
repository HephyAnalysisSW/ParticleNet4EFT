#    --data-test  '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/*.root'\
python train.py \
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/TT01jDebug_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctWIm.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobal_likelihoodFree.py\
    --model-prefix ctWIm_lin_quad\
    --batch-size 10  --start-lr 5e-3 --num-epochs 1 --optimizer ranger\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --tensorboard test\
    --gpus ""
