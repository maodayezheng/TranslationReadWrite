#$ -l h_rt=168:0:0
#$ -l tmem=15G
#$ -l h_vmem=15G
#$ -P gpu
#$ -l gpu=1,gpu_P100_16=1
#!/bin/sh

SOURCE="${BASH_SOURCE[0]}"
while [ -h "${SOURCE}" ]; do
  DIR="$( cd -P "$( dirname "${SOURCE}" )" && pwd )"
  SOURCE="$(readlink "${SOURCE}")"
  [[ ${SOURCE} != /* ]] && SOURCE="${DIR}/${SOURCE}"
done
DIR="$( cd -P "$( dirname "${SOURCE}" )" && pwd )"

OUT_DIR=code_outputs/$(date +%Y_%m_%d_%H_%M_%S)

mkdir ${OUT_DIR}

PYTHON_FILE=translation_experiment.py

cp ${PYTHON_FILE} ${OUT_DIR}/${PYTHON_FILE}

THEANO_FLAGS=mode=Mode,device=cuda0,floatX=float32 python -u ${PYTHON_FILE} ${DIR} ${OUT_DIR} | tee ${OUT_DIR}/out.txt