#!/bin/bash

gpu_id=0
time=`python ../utils/get_time.py`

# office31 -------------------------------------------------------------------------------------------------------------
for seed in 2020 2021 2022; do
    for src in  'webcam' 'amazon' 'dslr' ; do
        echo $src
        python DINE_dist.py --gpu_id $gpu_id --seed $seed --dset office31 --s $src --da uda --net_src resnet50 --max_epoch 50 --timestamp $time
        for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
            python DINE_dist_kDINE.py --gpu_id $gpu_id --seed $seed --dset office31 --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50  --distill --topk 1 --timestamp $time --pk_type ub --pk_uconf $pk_uconf
            python DINE_ft.py --gpu_id $gpu_id --seed $seed --dset office31 --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50 --lr 1e-2  --timestamp $time  --method kdine
        done

         python DINE_dist_kDINE.py --gpu_id $gpu_id --seed $seed --dset office31 --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50  --distill --topk 1 --timestamp $time --pk_type br --pk_uconf 1.0
         python DINE_ft.py --gpu_id $gpu_id --seed $seed --dset office31 --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50 --lr 1e-2  --timestamp $time  --method kdine
    done
done


# office-home ----------------------------------------------------------------------------------------------------------
for seed in 2020 2021 2022; do
    for src in 'Product' 'Real_World' 'Art' 'Clipart' ; do
        echo $src
        python DINE_dist.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da uda --net_src resnet50 --max_epoch 50 --timestamp $time
        for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
            python DINE_dist_kDINE.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50  --distill --topk 1 --timestamp $time --pk_type ub --pk_uconf $pk_uconf
            python DINE_ft.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50 --lr 1e-2  --timestamp $time --method kdine
        done

        python DINE_dist_kDINE.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50  --distill --topk 1 --timestamp $time --pk_type br --pk_uconf 1.0
        python DINE_ft.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da uda --net_src resnet50 --max_epoch 30 --net resnet50 --lr 1e-2  --timestamp $time --method kdine
    done
done

# office-home (PDA)-----------------------------------------------------------------------------------------------------
for seed in 2020 2021 2022; do
    for src in 'Product' 'Real_World' 'Art' 'Clipart' ; do
        echo $src
        python DINE_dist.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da pda --net_src resnet50 --max_epoch 50 --timestamp $time

        python DINE_dist_kDINE.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da pda --net_src resnet50 --max_epoch 30 --net resnet50  --distill --topk 1 --timestamp $time --pk_type ub --pk_uconf 0.0
        python DINE_dist_kDINE.py --gpu_id $gpu_id --seed $seed --dset office-home --s $src --da pda --net_src resnet50 --max_epoch 30 --net resnet50  --distill --topk 1 --timestamp $time --pk_type br --pk_uconf 1.0
    done
done