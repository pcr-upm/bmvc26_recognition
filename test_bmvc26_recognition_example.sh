#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t bmvc26_recognition_image .
sudo docker run --name bmvc26_recognition_container --ipc=host --rm --gpus all -it -d bmvc26_recognition_image bash
sudo docker exec -w /home/username/bmvc26_recognition bmvc26_recognition_container bash start.sh
sudo docker exec -e WANDB_MODE=disabled -w /home/username/bmvc26_recognition bmvc26_recognition_container torchrun --nnodes=1 --nproc_per_node=1 test/bmvc26_recognition_test.py --input-data test/example.mp4 --config configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py --ckpt data/thumos_vitb.pth --topk 10 --database thumos --gpu 0 --data-root test/ --save-video
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp bmvc26_recognition_container:/home/username/conda/envs/bmvc26/lib/python3.10/site-packages/pcr_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f bmvc26_recognition_container
sudo docker image rm bmvc26_recognition_image
sudo docker builder prune -a -f