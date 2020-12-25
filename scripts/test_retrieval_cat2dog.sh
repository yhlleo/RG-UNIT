GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_retrieval/get_images_embeddings.py \
  --config configs/cat2dog_retrieval_gmm.yaml \
  --mode gmm \
  --data_type Cat2Dog_retrieval_test \
  --json_name cat2dog_image_embeddings.json \
  --usage_mode train