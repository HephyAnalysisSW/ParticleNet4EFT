python train.py \
    --data-train '/scratch-cbe/users/lena.wild/tttt/nanoTuples/gen/v5/TTTT_MS/TTTT_MS_*.root'\
    --data-config    FourFermion/data/ak8_points_pf_full_weighttest_Lena.yaml\
    --network-config FourFermion/networks/particle_net_genjetAK8_weighttest_Lena.py\
    --model-prefix ctt_lin_quad\
    --batch-size 10  --start-lr 1e-4 --num-epochs 20 --optimizer ranger\
    --log FourFermion/logs/train_ctt.log\
    --regression-mode\
    --weighting\
    --tensorboard ctt\
    --gpus ""\
    #--load-epoch -1