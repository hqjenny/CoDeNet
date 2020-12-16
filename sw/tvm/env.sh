echo "TVM_PATH: ${TVM_PATH}"
echo "This path must be set in the environment."
export VTA_HW_PATH=$TVM_PATH/3rdparty/vta-hw
export PYTHONPATH=$TVM_PATH/vta/python:${PYTHONPATH}
export PYTHONPATH=$TVM_PATH/python:$TVM_PATH/topi/python:$TVM_PATH/nnvm/python:$TVM_PATH/vta/python:${PYTHONPATH}
# On the Host-side
export VTA_RPC_HOST=192.168.2.1
export VTA_RPC_PORT=9091
export VTA_ULTRA96_RPC_PORT=9091
export VTA_ULTRA96_RPC_HOST=192.168.2.1
