#    --in-memory\
ipython -i train.py --\
    --data-train '/scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/postprocessed/gen/v10/TT01j_HT800_ext_comb/TT01j_HT800_ext_comb_*.root'\
    --data-config    HadronicSMEFT/data/delphesJet_ctGIm.yaml\
    --network-config HadronicSMEFT/networks/ParticleNetGlobal_likelihoodFree.py\
    --model-prefix   /scratch-cbe/users/robert.schoefbeck/HadronicSMEFT/models/ctGIm/ctGIm\
    --start-lr 1e-2 --num-epochs 300 --optimizer adam --tensorboard "./ctGIm"\
    --batch-size 256\
    --steps-per-epoch 300\
    --log HadronicSMEFT/logs/train.log\
    --fetch-by-files\
    --fetch-step 10\
    --regression-mode\
    --weighting\
    --gpus 0\
    --load-epoch -1
