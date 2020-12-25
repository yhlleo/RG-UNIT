GPU_ID=$1
iters=$2
NUM_STYLE=$3

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 gmmunit_test.py \
  --config configs/cat2dog_gmmunit_retrieval.yaml \
  --checkpoint gen_${iters}0000.pt \
  --image_dir ../datasets/cat2dog \
  --test_list ./datasets/cat2dog/test_list.txt \
  --num_style ${NUM_STYLE}