checkpoint_path=/home/liupengli/myWork/FRE/checkpoints/fre.ckpt-42000

test_list=/home/liupengli/myWork/DataSets/HED-BSDS/across.lst

# python test.py \
python read_with_cv.py \
	--checkpoint_path=${checkpoint_path} \
    --test_list=${test_list}