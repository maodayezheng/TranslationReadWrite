#!/bin/sh

SOURCE="${BASH_SOURCE[0]}"
while [ -h "${SOURCE}" ]; do
  DIR="$( cd -P "$( dirname "${SOURCE}" )" && pwd )"
  SOURCE="$(readlink "${SOURCE}")"
  [[ ${SOURCE} != /* ]] && SOURCE="${DIR}/${SOURCE}"
done
DIR="$( cd -P "$( dirname "${SOURCE}" )" && pwd )"
source $SET_CUDA_DEVICE

OUT_DIR=code_outputs/$(date +%Y_%m_%d_%H_%M_%S)

mkdir ${OUT_DIR}

PYTHON_FILE=${1}

cp ${PYTHON_FILE} ${OUT_DIR}/${PYTHON_FILE}

THEANO_FLAGS=mode=Mode,device=cuda1,floatX=float32 python -u ${PYTHON_FILE} ${DIR} ${OUT_DIR} | tee ${OUT_DIR}/out.txt