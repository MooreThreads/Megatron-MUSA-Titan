export NCCL_PROTOS=2
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=10489 test_dist.py