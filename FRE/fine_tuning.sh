
checkpoint_path=/home/liupengli/myWork/FRE/checkpoints
tfrecord_path=/home/liupengli/myWork/FRE/FRE/tfrecord
checkpoint_dir=/home/liupengli/myWork/FRE/checkpoints
summary_dir=/home/liupengli/myWork/FRE/summaries
python train.py \
    --tfrecord_path=${tfrecord_path} \
    --checkpoint_dir=${checkpoint_dir} \
    --checkpoint_path=${checkpoint_path} \
    --summary_dir=${summary_dir} \
    --save_steps=100 \
	--optimizer='momentum' \
	--fine_tuning=True \
    --max_step=42000