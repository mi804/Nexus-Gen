# python train/decoder/download_flux.py

PYTHONPATH=$(pwd) \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train/decoder/train_flux_decoder.py --config train/configs/visual2image_full_81_512.yaml
