# tensorboard --logdir=outputs/dfine-hgnetv2-c2f-mfm-lsk/summary
# tensorboard --logdir=outputs/deim_hgnetv2_s_custom/summary
tensorboard --logdir=outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/summary
# tensorboard --logdir=outputs/dfine_hgnetv2_n_custom/summary
# python tools/benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_s_custom.yml
# python train.py -c configs/test/dfine_hgnetv2_test.yml --test-only -r outputs/dfine-hgnetv2-mfm-psconv/best_stg2.pth

# python train.py -c configs/test/dfine_hgnetv2_test.yml --test-only -r outputs/dfine-n-pconv-ema/best_stg2.pth
# python train.py -c configs/test/dfine_hgnetv2_test.yml --test-only -r outputs/dfine-n-c2f-mfm-lsk/best_stg2.pth
# python train.py -c configs/test/dfine_hgnetv2_test.yml --test-only -r outputs/dfine-n-lite-c2f-mfm-lsk-ema-v2/best_stg2.pth -u yolo_metrice=True
# python train.py -c configs/test/dfine_hgnetv2_test.yml --test-only -r outputs/dfine-n-mfm-lsk/best_stg2.pth -u yolo_metrice=True
# python train.py -c configs/dfine/dfine_hgnetv2_n_custom.yml --test-only -r outputs/dfine_hgnetv2_n_custom2/best_stg2.pth
# python train.py -c configs/deim/deim_hgnetv2_s_custom.yml --test-only -r outputs/deim_hgnetv2_s_custom/best_stg1.pth

## 推理
# python tools/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_s_custom.yml -r \
# outputs/dfine_hgnetv2_s_custom2/best_stg2.pth --input imgs --output inference_results/exp-dfine-s -t 0.4

# python tools/inference/torch_inf.py -c configs/test/dfine_hgnetv2_test.yml -r \
# outputs/dfine-hgnetv2-c2f-mfm-lsk/best_stg2.pth --input imgs --output inference_results/exp-c2f-mfm-lsk -t 0.4