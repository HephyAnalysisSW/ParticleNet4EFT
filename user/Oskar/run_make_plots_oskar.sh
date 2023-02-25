for model in models/*/
do
model=${model#models/}
model=${model%/}
echo $model
python make_plots_oskar.py --nr-files 0 --model-name $model
done