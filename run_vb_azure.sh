#!/bin/sh
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

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

THEANO_FLAGS=mode=Mode,device=cuda2,floatX=float32 python -u ${PYTHON_FILE} ${DIR} ${OUT_DIR} | tee ${OUT_DIR}/out.txt