#python re-id.py --gpus 0 --num-examples 119698 --batch-size 64 --lr 0.01 --num-epoches 100 --mode test-cls-verifi --train-file train --test-file test-800 --verifi --verifi-label
#python re-id.py --gpus 1 --num-examples 119698 --batch-size 64 --lr 0.01 --num-epoches 100 --mode test-cls --train-file train --test-file test-800 
python re-id.py --gpus 2 --num-examples 119698 --batch-size 64 --lr 0.01 --num-epoches 100 --mode test-cls-verifi-triplet --train-file train --test-file test-800 --verifi --verifi-label --triplet


