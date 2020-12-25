GPU_ID=$1
RESUME=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_gmmunit_retrieval.py \
  --config configs/cat2dog_gmmunit_retrieval.yaml \
  --resume ${RESUME}