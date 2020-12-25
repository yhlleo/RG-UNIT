GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} python attributes_classifier/eval_classifier.py \
  --config configs/celeba_faces_classifier_gmm.yaml \
  --model gmmunit
