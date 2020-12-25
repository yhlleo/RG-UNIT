GPU_ID=$1
iters=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 gmmunit_test.py \
  --config configs/celeba_faces_gmmunit_retrieval-0.50.yaml \
  --checkpoint gen_${iters}0000.pt