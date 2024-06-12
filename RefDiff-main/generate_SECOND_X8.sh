torchrun --standalone --nproc_per_node=8  generate.py \
	--sampler_stages=second \
        --batch=48 \
        --indir=/path/to/SECOND/RefSR_dataset/test/Bic_LR_D8_32 \
        --indir_ref=/path/to/SECOND/RefSR_dataset/test/Ref \
        --indir_mask=/path/to/SECOND/RefSR_dataset/test/CD_mask \
        --outdir=./experiments/results_SECOND_X8 \
	--seeds=0-1199 \
	--num_steps_second=256 \
        --network_second=/path/to/ckpt/SECOND_X8.pt


