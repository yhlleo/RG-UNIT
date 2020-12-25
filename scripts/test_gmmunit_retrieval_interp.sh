GPU_ID=$1
CHECKPOINTS=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 gmmunit_interp.py \
  --config configs/celeba_faces_gmmunit_retrieval.yaml \
  --checkpoint ${CHECKPOINTS} \
  --test_list ./valid/interp_demo.lst \
  --num_interp 16