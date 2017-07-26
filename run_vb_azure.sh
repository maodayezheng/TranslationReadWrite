#$ -l h_rt=168:0:0
#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -P gpu
#$ -l gpu=1,gpu_P100_16=1
#!/bin/sh

mkdir code_outputs/final_model

THEANO_FLAGS=mode=Mode,device=cuda0,floatX=float32 python -u translation_experiment.py code_outputs/final_model | tee code_outputs/final_model/out.txt