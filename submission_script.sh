#PBS -l select=1:ncpus=8:mem=96gb:ngpus=4:gpu_type=RTX6000
#PBS -l walltime=48:0:0

echo "Host - $HOSTNAME"
# echo "Commit - $(git rev-parse HEAD)"
nvidia-smi

# module load python/3.7
module purge
module load tools/dev
module load tools/prod
module load anaconda3/personal

source activate bias_ilql

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=/rds/general/user/bsk18/home/anaconda3/lib/

# pip install nvidia-cudnn-cu12

# Install dependencies.
cd $HOME/final-bias-ilql/scripts/train/hackernews/
# python $HOME/final-bias-ilql/scripts/train/hackernews/train_bc.py
# python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --use_env $HOME/final-bias-ilql/scripts/train/hackernews/train_bc.py
accelerate launch $HOME/final-bias-ilql/scripts/train/hackernews/train_bc.py


