GPU_ID=$1
RESUME=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python attributes_classifier/train.py \
  --config configs/celeba_faces_classifier.yaml \
  --resume ${RESUME} \
  --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Smiling Young Eyeglasses No_Beard