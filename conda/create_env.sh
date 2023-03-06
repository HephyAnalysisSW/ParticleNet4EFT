#!/bin/bash -x
#
# Setup GPU based environment
#
ENV_NAME="test-weaver"


. /software/2020/software/mamba/22.11.1-4/etc/profile.d/conda.sh
. /software/2020/software/mamba/22.11.1-4/etc/profile.d/mamba.sh

if [ "$CONDA_DEFAULT_ENV" != "base" ]
then
    conda activate base
fi

# root has to be installed in a seperate step from the rest due to bug in install scriptlet
mamba create -y -n "$ENV_NAME" python=3.9 root root_numpy
if [ $? -ne 0 ]
then
  echo "mamba failed"
  exit 1
fi

mamba install -y -n "$ENV_NAME" -c pytorch -c nvidia --file=environment.txt
if [ $? -ne 0 ]
then
  echo "mamba failed"
  exit 1
fi

conda activate "$ENV_NAME"

pip install --upgrade pip
pip install -r requirements.txt
