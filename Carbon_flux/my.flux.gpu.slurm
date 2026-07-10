#!/bin/bash
#SBATCH --job-name=flux_gpu # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem 507904M # in unit MB 507904M=496G; is the max of the innovator 
#SBATCH --gres=gpu:2   # Number of GPUs per node (MAX is 2 for SDSU tested on Hank on Apr 19 2021)
#SBATCH --ntasks-per-node=40 # CPUs per node (MAX=40 for CPU nodes and 80 for GPU)
#SBATCH --output=out-%j-%N.log # Standard output (log file)
#SBATCH --partition=gpu # Partition/Queue
#SBATCH --time=5-00:00:00 # Maximum walltime 192=8-days
#SBATCH --mail-type=ALL #see https://slurm.schedmd.com/sbatch.html for all options.
#SBATCH --mail-user=hankui.zhang@sdstate.edu

# cd /mmfs1/home/jacks.local/hankui.zhang/tensorflow2/flux_net/Carbon_flux_20240501
# sbatch my.flux.gpu.slurm
# squeue -l | grep hankui
# sleep 18000

# nvidia-smi

# module load rasterio/1.3.9
module load rasterio/tensorflow # changed on Mar 5, 2024 
# conda init bash 
# date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
which python

rm -rf ./tmp/tb_log/*


SLEEP=60
MAXGPU=1; versions="5_1"; epochs="100" # no pre-training 
MAXGPU=1; versions="5_2"; epochs="100" # good one with 64 
# MAXGPU=1; versions="5_4"; epochs="100" # no pre-training 
[ -n "$1" ] && versions=${1}

date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
date

gpui=0
layer=4
year=1985
year=""
for learning_rate in 0.01;
# for learning_rate in 0.01 0.001;  ## transformer models 
do
for L2 in 0.0001;
do
for epoch in ${epochs};
# for epoch in 20;
do
for BATCH in 16;
# for BATCH in 16 32 64;
do
# for DROP in 0 0.1 0.2 0.3 0.4 0.5;
for DROP in 0.1;
# for DROP in 0.3;
# for DROP in 0.05 0.1;
do
for version in ${versions};
# for version in 12_52;
do
	# learning_rate=0.0001; method=1; epoch=70; L2=0; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	# learning_rate=0.0001; method=2; L2=1e-5; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	method=2; sleep ${SLEEP}; ## Hank layer=5; perc=0.1; 
	echo "python Pro_2d1d_CNN_v${version}.py ${learning_rate} ${epoch} ${method} ${L2} ${BATCH} 0 ${year} "
	python Pro_cnn_transformer_mul_cross_HK_v${version}.py ${epoch} ${learning_rate} ${BATCH} ${L2} > flux.DROP${DROP}.rate${learning_rate}.b${BATCH}.e${epoch}.L${L2}.v${version}.txt${year} & 
	gpui=$((${gpui}+1))
	if [ ${gpui} -ge ${MAXGPU} ]; then 
		wait 
		gpui=0
		echo "gpui=0"
	fi
# done 
done 
done
done 
done 
done 
done

wait 
date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
date
echo ${date_start}
echo ${date_end}
echo "$time_diff hours used";







