#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1:gpu_type=A100

echo "Host - $HOSTNAME"

# echo "Commit - $(git rev-parse HEAD)"
# nvidia-smi
# nvidia-smi nvlink -c

# module load python/3.7
module purge
module load git/2.41.0-GCCcore-12.3.0-nodocs
module load git-lfs/3.2.0 
# module load tools/dev
# module load tools/prod
# module load anaconda3/personal
eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda activate fms_at_work
# pip install torch torchvision torchaudio
# pip install transformers


echo "Host - $HOSTNAME"
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
# MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
# MASTER_PORT=$(shuf -i 2000-65000 -n 1)
# echo MASTER_ADDR= $MASTER_ADDR
# echo MASTER_PORT= $MASTER_PORT

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=$HOME/miniconda3/lib/
export NCCL_DEBUG=INFO
# export NCCL_BLOCKING_WAIT=1
# # export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=5400000  # Setting a longer timeout
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export TOKENIZERS_PARALLELISM=false
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

# H=`hostname`
# THEID=`echo -e $HOSTNAMES  | python -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
# echo THEID=$THEID
# pip install nvidia-cuda-nvcc-cu12
# pip uninstall flash-attn -y
# cd $HOME/FMs-at-work/flash-attention
# python setup.py install
# pip uninstall nvidia-cudnn-cu12 -y
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# pip install --use-pep517 flash-attn --no-build-isolation

# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install nvidia-cudnn-cu12==8.9.2.26
# pip install torch torchvision torchaudio --upgrade
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install dependencies.
cd $HOME/FMs-at-work/
python /gpfs/home/bsk18/FMs-at-work/scripts/eval/hackernews/inference_policy.py
# accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_bc.py
# accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_iql.py model.load.checkpoint_path=/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/lrl1e-5-512-working/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.4 train.save_checkpoint_dir=outputs/hackernews/llama/frozen_512_bios_no_offload_5_epoch_awac0-4
# accelerate launch $HOME/FMs-at-work/scripts/eval/hackernews/eval_policy.py model.load.checkpoint_path=outputs/hackernews/llama/frozen_512_bios_no_offload_5_epoch_awac0-4/model_7499.pkl eval.log_save_path=outputs/hackernews/llama/eval/frozen_512_hn_no_offload_5_epoch-beta8_awac04/eval_logs.pkl

# accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_iql.py model.load.checkpoint_path=/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/lrl1e-5-512-working/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.4 train.save_checkpoint_dir=outputs/hackernews/llama/frozen_512_bios_no_offload_5_epoch_awac0-4
# accelerate launch $HOME/FMs-at-work/scripts/eval/hackernews/eval_policy.py model.load.checkpoint_path=outputs/hackernews/llama/frozen_512_bios_no_offload_5_epoch_awac0-4/model_10499.pkl eval.log_save_path=outputs/hackernews/llama/eval/frozen_512_hn_no_offload_5_epoch-beta8_awac04/eval_logs.pkl

# accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_iql.py model.load.checkpoint_path=/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/lrl1e-5-512-working/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.00 train.save_checkpoint_dir=outputs/hackernews/llama/frozen_512_10_epoch_awac0-00_w_eval train.epochs=10
# accelerate launch 
# python $HOME/FMs-at-work/scripts/eval/hackernews/eval_policy.py model.load.checkpoint_path=outputs/hackernews/llama/frozen_512_bios_no_offload_10_epoch_awac0-25/model_11999_backup.pkl eval.log_save_path=outputs/hackernews/llama/eval/frozen_512_hn_no_offload_10_epoch-beta64_awac025_1gpu/eval_logs_16499.pkl

# accelerate launch $HOME/FMs-at-work/scripts/eval/hackernews/eval_policy.py model.load.checkpoint_path=outputs/hackernews/llama/frozen_512_bios_no_offload_10_epoch_awac0-25/model_16499.pkl eval.log_save_path=outputs/hackernews/llama/eval/frozen_512_hn_no_offload_10_epoch-beta8_awac025/eval_logs_16499.pkl

# accelerate launch $HOME/FMs-at-work/scripts/train/hackernews/train_iql.py model.load.checkpoint_path=/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/lrl1e-5-512-working/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.9 train.save_checkpoint_dir=outputs/hackernews/llama/frozen_512_bios_no_offload_10_epoch_awac0-9 train.epochs=10
# accelerate launch $HOME/FMs-at-work/scripts/eval/hackernews/eval_policy.py model.load.checkpoint_path=outputs/hackernews/llama/frozen_512_bios_no_offload_10_epoch_awac0-9/model_22499.pkl eval.log_save_path=outputs/hackernews/llama/eval/frozen_512_hn_no_offload_10_epoch-beta8_awac09/eval_logs.pkl


# python /gpfs/home/bsk18/FMs-at-work/data/hackernews_rl_dataset/llama_prompts.py
# python $HOME/FMs-at-work/scripts/eval/hackernews/push_to_hf.py
# python $HOME/FMs-at-work/data/hackernews_rl_dataset/hackernews_occup_label.py $PBS_ARRAY_INDEX
# python $HOME/FMs-at-work/scripts/eval/hackernews/impact_ratio_calc_hackernews.py $PBS_ARRAY_INDEX
# python $HOME/FMs-at-work/scripts/eval/hackernews/process_biasinbios.py $PBS_ARRAY_INDEX
# python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --use_env $HOME/final-bias-ilql/scripts/train/hackernews/train_bc.py
# accelerate launch --machine_rank $PBS_TASKNUM --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_processes 2 --num_machines 2 --multi_gpu --mixed_precision fp16 $HOME/FMs-at-work/scripts/train/hackernews/train_bc.py 
# python $HOME/FMs-at-work/scripts/train/hackernews/train_dpo.py model.load.checkpoint_path=$HOME/FMs-at-work/outputs/hackernews/gpt2/lrl1e-4-20.pkl/model.pkl
# python /gpfs/home/bsk18/FMs-at-work/data/hackernews_rl_dataset/hackernews_occup_label.py 50
# torchrun --nproc_per_node 1 --nnodes 3 $HOME/FMs-at-work/scripts/train/hackernews/train_iql.py model.load.checkpoint_path=/gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/lrl1e-5-512-working/model_converted.pkl model.load.strict_load=false train.loss.awac_weight=0.0
# python  $HOME/FMs-at-work/scripts/eval/hackernews/distill_policy_eval.py --eval_file /gpfs/home/bsk18/FMs-at-work/outputs/hackernews/llama/eval/frozen_512_bios_no_cpu_offload-beta8_400/eval_logs.pkl
# python $HOME/FMs-at-work/benchmarking/sentence-debiasing/sentence_debias.py --mode mistral --index $PBS_ARRAY_INDEX
# python $HOME/FMs-at-work/benchmarking/diversity_benchmark.py --index $PBS_ARRAY_INDEX --method mistral
# python $HOME/FMs-at-work/scripts/eval/hackernews/impact_ratio_calc.py --save_path beta4-openllama
# python $HOME/FMs-at-work/benchmarking/diversity_benchmark.py
# python $HOME/FMs-at-work/benchmarking/language_quality_benchmark_autorefine.py