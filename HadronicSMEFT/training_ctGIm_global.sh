#    --data-test  '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/*.root'\

python train.py \
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v9/TT01jDebug/TT01jDebug_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobal_likelihoodFree.py\
    --model-prefix ctGIm_lin_quad\
    --batch-size 10  --start-lr 1e-4 --num-epochs 20 --optimizer ranger\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --tensorboard test\
    --gpus ""\
    --load-epoch -1
