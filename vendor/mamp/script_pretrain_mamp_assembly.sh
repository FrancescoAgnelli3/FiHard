export OMP_NUM_THREADS=1

# Optional: set this manually
# export CUDA_VISIBLE_DEVICES=0

python3 main_pretrain.py \
  --config ./config/assembly_joint/pretrain_mamp_t70_layer8+5_mask90.yaml \
  --output_dir ./output_dir/assembly_joint/pretrain_mamp_t70_layer8+5_mask90_tau0.80_ep400 \
  --log_dir ./output_dir/assembly_joint/pretrain_mamp_t70_layer8+5_mask90_tau0.80_ep400
