python3 -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --with_seg_net=False --with_decision_net=True --storage_dir=silverplot --dataset_dir=db --datasets=SilverPlot --name_prefix=decision-net_full-size_cross-entropy --pretrained_main_folder=silverplot/segdec_train/SilverPlot/full-size_cross-entropy/
