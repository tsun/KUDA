#!/bin/bash

gpu_id=0
time=`python ../util/get_time.py`

# office31 -------------------------------------------------------------------------------------------------------------
for src in "amazon" "webcam" "dslr"; do
    echo $src
    python image_source.py --trte val --da uda --gpu_id $gpu_id --dset office31 --s $src --max_epoch 100 --timestamp $time
done

for seed in 2020 2021 2022; do
    for src in "amazon" "webcam" "dslr"; do
        echo $src
        for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
            python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset office31 --s $src --timestamp $time --seed $seed --pk_uconf $pk_uconf --pk_type ub
        done
        python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset office31 --s $src --timestamp $time --seed $seed --pk_uconf 1.0 --pk_type br
    done
done


# office-home-rsut ----------------------------------------------------------------------------------------------------
for src in "Product" "Clipart" "Real_World"; do
    echo $src
    python image_source.py --trte val --da uda --gpu_id $gpu_id --dset office-home-rsut --s $src --max_epoch 50 --timestamp $time
done

for seed in 2020 2021 2022; do
    for src in "Product" "Clipart" "Real_World"; do
        echo $src
        for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
            python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset office-home-rsut --s $src --timestamp $time --seed $seed --pk_uconf $pk_uconf --pk_type ub
        done
        python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset office-home-rsut --s $src --timestamp $time --seed $seed --pk_uconf 1.0 --pk_type br
    done
done


# office-home ----------------------------------------------------------------------------------------------------------
for src in "Product" "Clipart" "Art" "Real_World"; do
    echo $src
    python image_source.py --trte val --da uda --gpu_id $gpu_id --dset office-home --s $src --max_epoch 50 --timestamp $time
done

for seed in 2020 2021 2022; do
    for src in "Product" "Clipart" "Art" "Real_World"; do
        echo $src
        for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
            python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset office-home --s $src --timestamp $time --seed $seed --pk_uconf $pk_uconf --pk_type ub
        done
        python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset office-home --s $src --timestamp $time --seed $seed --pk_uconf 1.0 --pk_type br
    done
done


# visda-2017 -----------------------------------------------------------------------------------------------------------
python image_source.py --trte val --da uda --gpu_id $gpu_id --dset visda-2017 --s train --max_epoch 10 --timestamp $time --net resnet101 --lr 1e-3

for seed in 2020 2021 2022; do
    for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
        python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset visda-2017 --s train --timestamp $time --seed $seed --pk_uconf $pk_uconf --net resnet101 --lr 1e-3 --pk_type ub
    done
    python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset visda-2017 --s train --timestamp $time --seed $seed --pk_uconf 1.0 --net resnet101 --lr 1e-3 --pk_type br
done


# domainnet40 ----------------------------------------------------------------------------------------------------------
for src in "sketch" "clipart" "painting" "real"; do
    echo $src
    python image_source.py --trte val --da uda --gpu_id $gpu_id --dset domainnet40 --s $src --max_epoch 50 --timestamp $time
done

for seed in 2020 2021 2022; do
    for src in "sketch" "clipart" "painting" "real"; do
        echo $src
        for pk_uconf in 0.0 0.1 0.5 1.0 2.0; do
            python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset domainnet40 --s $src --timestamp $time --seed $seed --pk_uconf $pk_uconf --pk_type ub
        done
        python image_target_kSHOT.py --cls_par 0.3 --da uda --gpu_id $gpu_id --dset domainnet40 --s $src --timestamp $time --seed $seed --pk_uconf 1.0 --pk_type br
    done
done


# office-home (PDA)-----------------------------------------------------------------------------------------------------
for src in "Product" "Clipart" "Art" "Real_World"; do
    echo $src
    python image_source.py --trte val --da pda --gpu_id $gpu_id --dset office-home --s $src --max_epoch 50 --timestamp $time
done

for seed in 2020 2021 2022; do
    for src in "Product" "Clipart" "Art" "Real_World"; do
        echo $src
        python image_target_kSHOT.py --cls_par 0.3 --da pda --gpu_id $gpu_id --dset office-home --s $src --timestamp $time --seed $seed --pk_uconf 0.0 --pk_type ub
        python image_target_kSHOT.py --cls_par 0.3 --da pda --gpu_id $gpu_id --dset office-home --s $src --timestamp $time --seed $seed --pk_uconf 1.0 --pk_type br
    done
done