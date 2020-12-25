GPU_ID=$1
CONFIG=$2
CHECKPOINT=$3

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 gmmunit_demo.py \
  --config ${CONFIG} \
  --test_list valid/demo2.lst \
  --checkpoint ${CHECKPOINT}