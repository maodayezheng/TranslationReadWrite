#$ -l h_rt=168:0:0
#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -P gpu
#$ -l gpu=1,gpu_P100_16=1
#!/bin/sh

OUT_DIR=code_outputs/final_model

mkdir ${OUT_DIR}

PYTHON_FILE=translation_experiment.py

cp ${PYTHON_FILE} ${OUT_DIR}/${PYTHON_FILE}

THEANO_FLAGS=mode=Mode,device=cuda0,floatX=float32 python -u ${PYTHON_FILE} ${OUT_DIR} | tee ${OUT_DIR}/out.txt