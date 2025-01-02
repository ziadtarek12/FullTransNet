echo "============================================"
echo "Traing summe"
echo "============================================"

CUDA_VISIBLE_DEVICES=1 python3 train.py --model-dir ./model_save/model_full --splits ./splits/tvsum.yml --dff 2048 --seed 2021 --max-epoch 300 --num_head 8  --enlayers 6 --delayers 6 --local_layers 0 --keyframes_type shot  --length 1536 --window-size 16 --lr 0.0013  --weight_decay 5e-6 --dim_mid 64 --log-file logfull.txt