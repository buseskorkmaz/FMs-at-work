#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=10:mem=320gb:ngpus=4:gpu_type=A100

echo "Host - $HOSTNAME"
# echo "Commit - $(git rev-parse HEAD)"
nvidia-smi

# module load python/3.7
module purge
module load git/2.41.0-GCCcore-12.3.0-nodocs
module load git-lfs/3.2.0 
# module load tools/dev
# module load tools/prod
# module load anaconda3/personal
eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda activate bias_ilql

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=$HOME/miniconda3/lib/

# pip install nvidia-cudnn-cu12
# pip install flash-attn

# Install dependencies.
cd $HOME/FMs-at-work/scripts/train/hackernews/
# python $HOME/final-bias-ilql/scripts/train/hackernews/train_bc.py
# python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --use_env $HOME/final-bias-ilql/scripts/train/hackernews/train_bc.py
# accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_bc.py
accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_iql.py model.load.checkpoint_path=/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/openllama/lrl1e-6-10.pkl/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.0

