#!/bin/bash
#SBATCH --job-name=cvd_thesis_hpc
#SBATCH --output=cvd_hpc_%j.log
#SBATCH --error=cvd_hpc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32              # Adjust to match your cluster allocation (e.g. 16, 32, 64)
#SBATCH --mem=32G                       # Adjust memory as needed
#SBATCH --time=04:00:00                 # Maximum walltime
#SBATCH --partition=cpu                 # Adjust partition/queue name for your cluster

echo "=========================================================="
echo "Starting CVD Training & Testing Pipeline on HPC CPUs"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node Allocated: $SLURM_NODELIST"
echo "CPUs per task : $SLURM_CPUS_PER_TASK"
echo "Start Time    : $(date)"
echo "=========================================================="

# 1. Threading & oneDNN Vectorization Optimization
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-32}
export TF_NUM_INTEROP_THREADS=4
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,1,0
export TF_ENABLE_ONEDNN_OPTS=1

# 2. Activate Python Environment (adjust module/venv path as needed)
# module load python/3.11
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 3. Execute Full Deep Learning & Digital Twin Pipeline
python run_pipeline.py \
    --epochs 100 \
    --batch-size 512 \
    --out-dir cvd_thesis_results_hpc \
    --shap-samples 150 \
    --shap-background 75

echo "=========================================================="
echo "HPC Pipeline Execution Finished at: $(date)"
echo "=========================================================="
