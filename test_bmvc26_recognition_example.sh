#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t bmvc26_recognition_image .
sudo docker run --name bmvc26_recognition_container --ipc=host --rm --gpus all -it -d bmvc26_recognition_image bash
sudo docker exec -w /home/username/bmvc26_recognition bmvc26_recognition_container python test/bmvc26_recognition_latent.py --database shapes3d --gpu 0 --nlats 6 --ndims 4
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp bmvc26_recognition_container:/home/username/conda/envs/bmvc26/lib/python3.10/site-packages/pcr_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f bmvc26_recognition_container
sudo docker image rm bmvc26_recognition_image
sudo docker builder prune -a -f