for script in *.sbatch
do
echo $script
sbatch $script
done