#    --data-test  '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v8/TT01jDebug/*.root'\

python train.py \
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v9/TT01jDebug/TT01jDebug_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGRe.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobal_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGRe-v2/ctGRe\
    --batch-size 128  --start-lr 5e-4 --num-epochs 20 --optimizer ranger\
    --log HadronicSMEFT/logs/train.log\
    --regression-mode\
    --weighting\
    --tensorboard test\
    --gpus 0\
    --load-epoch -1
