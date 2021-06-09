
NUM_GPUS=1

python3 train_retrieval.py \
  --world_size ${NUM_GPUS} \
  --num_gpus ${NUM_GPUS} \
  --config_path configs/retrieval_celeba.yaml \
  --image_dir datasets/celeba/images \
  --train_list_path datasets/celeba/list_attr_celeba-train.txt \
  --test_list_path datasets/celeba/list_attr_celeba-val.txt \
  --checkpoint_dir expr/retrieval_celeba_checkpoint \
  --pretrained_path pretrained_models/gmmunit_gen.pth