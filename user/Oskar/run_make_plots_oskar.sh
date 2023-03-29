cd ~/CMS/ParticleNet4EFT/TopDecay
for model in models/*/
do
model=${model#models/}
model=${model%/}
echo $model
python ~/CMS/ParticleNet4EFT/user/oskar/make_plots_oskar.py --model-name $model
done