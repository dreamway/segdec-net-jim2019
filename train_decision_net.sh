python3 -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --size_height=1408 --size_width=512 --with_seg_net=False --with_decision_net=True --storage_dir=output --dataset_dir=db --datasets=output --name_prefix=decision-net_full-size_cross-entropy --pretrained_main_folder=output/segdec_train/output/full-size_cross-entropy/