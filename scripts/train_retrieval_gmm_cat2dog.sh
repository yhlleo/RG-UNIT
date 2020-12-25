GPU_ID=$1
RESUME=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_retrieval_gmm_cat2dog.py \
  --config configs/cat2dog_retrieval_gmm.yaml \
  --resume ${RESUME}
