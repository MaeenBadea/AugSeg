tport=52009
ngpu=1
ROOT=.

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch \
    --nproc_per_node=${ngpu} \
    --node_rank=0 \
    --master_port=${tport} \
    $ROOT/train_semi.py \
    --config=$ROOT/exps/mrun_vocs/voc_semi732_r50/config_semi.yaml --seed 2 --port ${tport}
