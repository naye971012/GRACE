
export OMP_NUM_THREADS=1

#deepspeed for fast training
torchrun --nproc_per_node=2 --nnode=1 \
         --node_rank=0 --master_addr=lab3 \
         --master_port=9901 train.py \
         --deepspeed ./scripts/deepspeed_config.json