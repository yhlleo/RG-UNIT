GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_retrieval/get_queries_embeddings.py \
  --config configs/celeba_faces_retrieval_gmm.yaml \
  --mode gmm
  #--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_retrieval/get_images_embeddings.py \
  --config configs/celeba_faces_retrieval_gmm.yaml \
  --mode gmm
  #--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
  
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_retrieval/evaluate_retrieval.py  \
  --config configs/celeba_faces_retrieval_gmm.yaml
