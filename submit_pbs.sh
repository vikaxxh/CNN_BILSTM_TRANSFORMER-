#!/bin/bash
#PBS -N cvd_thesis_pbs
#PBS -o cvd_pbs.log
#PBS -e cvd_pbs.err
#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=04:00:00
#PBS -q batch

# Change to the submission directory in PBS
if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
fi

echo "=========================================================="
echo "Starting CVD Training & Testing Pipeline on PBS HPC"
echo "Job ID        : $PBS_JOBID"
echo "Working Dir   : $(pwd)"
echo "Start Time    : $(date)"
echo "=========================================================="

# 1. Threading & oneDNN Vectorization Optimization
export OMP_NUM_THREADS=32
export TF_NUM_INTRAOP_THREADS=32
export TF_NUM_INTEROP_THREADS=4
export KMP_BLOCKTIME=0
export TF_ENABLE_ONEDNN_OPTS=1

# 2. Activate Python Environment (Virtualenv or Conda)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 3. Execute Pipeline
python run_pipeline.py \
    --epochs 100 \
    --batch-size 512 \
    --out-dir cvd_thesis_results_hpc \
    --shap-samples 150 \
    --shap-background 75

echo "=========================================================="
echo "PBS HPC Pipeline Finished at: $(date)"
echo "=========================================================="
