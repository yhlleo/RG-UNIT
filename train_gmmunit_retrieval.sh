
NUM_GPUS=1

python3 train_gmmunit.py \
  --world_size ${NUM_GPUS} \
  --num_gpus ${NUM_GPUS} \
  --config_path configs/gmmunit_celeba.yaml \
  --image_dir datasets/celeba/images \
  --train_list_path datasets/celeba/list_attr_celeba-train.txt \
  --test_list_path datasets/celeba/list_attr_celeba-val.txt \
  --vgg_model_path pretrained_models/vgg16-397923af.pth \
  --sample_dir expr/gmmunit_retrieval_celeba_sample \
  --checkpoint_dir expr/gmmunit_retrieval_celeba_checkpoint \
  --img_embedding_path pretrained_models/image_embeddings.pth \
  --pretrained_gen_path pretrained_models/gmmunit_gen.pth\
  --pretrained_dis_path pretrained_models/gmmunit_dis.pth\
  --pretrained_ret_path pretrained_models/gmmunit_ret.pth