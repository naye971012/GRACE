https://huggingface.co/docs/transformers/main/deepspeed

   64.35GB |   0.38GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   64.35GB |   0.38GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   57.20GB |   2.77GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   57.20GB |   2.77GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    1.15GB |  21.83GB | offload_param=none, offload_optimizer=none, zero_init=1
   28.60GB |  21.83GB | offload_param=none, offload_optimizer=none, zero_init=0


Fastest	Memory efficient
ZeRO-1	ZeRO-3 + offload
ZeRO-2	ZeRO-3
ZeRO-2 + offload	ZeRO-2 + offload
ZeRO-3	ZeRO-2
ZeRO-3 + offload	ZeRO-1




eval/R@1 0.82939
wandb:               eval/R@10 0.96844
wandb:                eval/R@5 0.95957
# GRACE
Generative Cross-modal Retrieval Reproducing...
