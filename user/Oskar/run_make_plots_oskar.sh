cd ~/CMS/ParticleNet4EFT

# for model in models/test_pnet_update_17_04/*/
# do
# model=${model#models/}
# model=${model%/}
# echo $model
# python ~/CMS/ParticleNet4EFT/user/Oskar/make_plots_oskar.py --model-name $model
# done


model=pre_train_fc_pnet
for epoch in 0 399 400 429 430 459
do

python ~/CMS/ParticleNet4EFT/user/Oskar/make_plots_oskar.py --model-name $model --epoch $epoch

done