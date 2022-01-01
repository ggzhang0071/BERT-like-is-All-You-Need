#docker_cmd_93.sh
img="nvcr.io/nvidia/pytorch:21.03-py3"

#img="padim:0.1"


docker run --gpus all  --privileged=true   --workdir /git --name "bert_like"  -e DISPLAY --ipc=host -d --rm  -p 6604:4452  \
-v /raid/git/BERT-like-is-All-You-Need:/git/BERT-like-is-All-You-Need \
 -v /raid/git/datasets:/git/datasets \
 $img sleep infinity

docker exec -it bert_like /bin/bash

pip list  |grep "pytorch"
