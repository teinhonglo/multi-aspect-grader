export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export WANDB_MODE=offline

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate w2v-grader
