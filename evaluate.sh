echo "============================================"
echo "evaluate summe:"
echo "============================================"

CUDA_VISIBLE_DEVICES=2 python3 evaluate.py --model-dir ./model_save/model_full --splits ./splits/summe.yml --dff 2048 --seed 2021 --max-epoch 300 --num_head 8  --enlayers 6 --delayers 6 --local_layers 0 --keyframes_type shot  --length 1536 --window-size 16 --lr 0.0013    --weight_decay 5e-6 --log-file log_summe_full_dff2048.txt