GPU_ID=$1
RESUME=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_retrieval_gmm.py \
  --config configs/celeba_faces_retrieval_gmm.yaml \
  --resume ${RESUME}
  #--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young