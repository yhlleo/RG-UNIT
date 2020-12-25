GPU_ID=$1
RESUME=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_gmmunit_retrieval.py \
  --config configs/celeba_faces_gmmunit_retrieval-0.50.yaml \
  --resume ${RESUME}