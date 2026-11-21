python src/train.py \
    --data_dir ./data/iclevr \
    --train_json ./train.json \
    --objects_json ./objects.json \
    --output_dir ./checkpoints \
    --epochs 150 \
    --batch_size 64 \
    --lr 1e-4 \
    --guidance_scale 3.5

# 推論、產生所有必要圖片並評估準確率
python src/sample.py \
    --ckpt ./checkpoints/best_model.pth \
    --test_json ./test.json \
    --new_test_json ./new_test.json \
    --objects_json ./objects.json \
    --output_dir ./images \
    --guidance_scale 3.5 \
    --num_inference_steps 100 \
    --sampler ddpm