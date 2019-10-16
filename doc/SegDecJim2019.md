



# SegDec Jim2019 code practice



- Step 1, train segmentation net

  ```bash
  python3 segdec_train.py  --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_typ=ENTROPY --size_height=1408 --size_width=512 --with_seg_net=True --with_decision_net=False --storage_dir=output --dataset_dir=db --datasets=kolektorSDD-dilate=5 --name_prefix=full-size_cross-entropy
  
  ```

  the results:

  ```bash
  2019-09-11 18:06:11.314856: step 6116, loss = 0.00199 (2.9 examples/sec; 0.348 sec/batch)
  2019-09-11 18:06:15.130687: step 6127, loss = 0.01137 (2.9 examples/sec; 0.345 sec/batch)
  2019-09-11 18:06:18.951447: step 6138, loss = 0.00204 (2.9 examples/sec; 0.348 sec/batch)
  2019-09-11 18:06:22.759881: step 6149, loss = 0.00970 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:06:28.021822: step 6160, loss = 0.00200 (2.9 examples/sec; 0.343 sec/batch)
  2019-09-11 18:06:31.866216: step 6171, loss = 0.00835 (2.9 examples/sec; 0.350 sec/batch)
  2019-09-11 18:06:35.677787: step 6182, loss = 0.00195 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:06:39.486613: step 6193, loss = 0.01018 (2.9 examples/sec; 0.345 sec/batch)
  2019-09-11 18:06:43.308443: step 6204, loss = 0.00198 (2.9 examples/sec; 0.349 sec/batch)
  2019-09-11 18:06:47.116146: step 6215, loss = 0.00963 (2.9 examples/sec; 0.344 sec/batch)
  2019-09-11 18:06:50.933156: step 6226, loss = 0.00196 (2.9 examples/sec; 0.346 sec/batch)
  2019-09-11 18:06:54.742148: step 6237, loss = 0.00955 (2.9 examples/sec; 0.346 sec/batch)
  2019-09-11 18:06:58.566576: step 6248, loss = 0.00199 (2.9 examples/sec; 0.346 sec/batch)
  2019-09-11 18:07:02.380951: step 6259, loss = 0.01013 (2.8 examples/sec; 0.351 sec/batch)
  2019-09-11 18:07:07.572684: step 6270, loss = 0.00193 (2.9 examples/sec; 0.348 sec/batch)
  2019-09-11 18:07:11.401013: step 6281, loss = 0.01324 (2.9 examples/sec; 0.349 sec/batch)
  2019-09-11 18:07:15.204913: step 6292, loss = 0.00195 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:07:19.019974: step 6303, loss = 0.00531 (2.9 examples/sec; 0.343 sec/batch)
  2019-09-11 18:07:22.827379: step 6314, loss = 0.00192 (2.9 examples/sec; 0.345 sec/batch)
  2019-09-11 18:07:26.641442: step 6325, loss = 0.00975 (2.9 examples/sec; 0.346 sec/batch)
  2019-09-11 18:07:30.446963: step 6336, loss = 0.00198 (2.9 examples/sec; 0.345 sec/batch)
  2019-09-11 18:07:34.268687: step 6347, loss = 0.01083 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:07:38.087536: step 6358, loss = 0.00187 (2.9 examples/sec; 0.344 sec/batch)
  2019-09-11 18:07:41.905306: step 6369, loss = 0.00919 (2.9 examples/sec; 0.349 sec/batch)
  2019-09-11 18:07:46.995763: step 6380, loss = 0.00193 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:07:50.821940: step 6391, loss = 0.00807 (2.9 examples/sec; 0.350 sec/batch)
  2019-09-11 18:07:54.628787: step 6402, loss = 0.00189 (2.9 examples/sec; 0.342 sec/batch)
  2019-09-11 18:07:58.440996: step 6413, loss = 0.00528 (2.9 examples/sec; 0.346 sec/batch)
  2019-09-11 18:08:02.253497: step 6424, loss = 0.00194 (2.9 examples/sec; 0.348 sec/batch)
  2019-09-11 18:08:06.074368: step 6435, loss = 0.00812 (2.9 examples/sec; 0.344 sec/batch)
  2019-09-11 18:08:09.882197: step 6446, loss = 0.00189 (2.9 examples/sec; 0.349 sec/batch)
  2019-09-11 18:08:13.697866: step 6457, loss = 0.00994 (2.9 examples/sec; 0.344 sec/batch)
  2019-09-11 18:08:17.512467: step 6468, loss = 0.00191 (2.9 examples/sec; 0.343 sec/batch)
  2019-09-11 18:08:21.320914: step 6479, loss = 0.01051 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:08:26.458286: step 6490, loss = 0.00188 (2.9 examples/sec; 0.346 sec/batch)
  2019-09-11 18:08:30.279474: step 6501, loss = 0.00598 (2.9 examples/sec; 0.348 sec/batch)
  2019-09-11 18:08:34.089370: step 6512, loss = 0.00187 (2.9 examples/sec; 0.344 sec/batch)
  2019-09-11 18:08:37.906043: step 6523, loss = 0.00876 (2.9 examples/sec; 0.350 sec/batch)
  2019-09-11 18:08:41.704271: step 6534, loss = 0.00191 (2.9 examples/sec; 0.343 sec/batch)
  2019-09-11 18:08:45.519833: step 6545, loss = 0.01123 (2.9 examples/sec; 0.345 sec/batch)
  2019-09-11 18:08:49.339180: step 6556, loss = 0.00187 (2.9 examples/sec; 0.350 sec/batch)
  2019-09-11 18:08:53.155955: step 6567, loss = 0.01232 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:08:56.974353: step 6578, loss = 0.00188 (2.9 examples/sec; 0.347 sec/batch)
  2019-09-11 18:09:00.783905: step 6589, loss = 0.00918 (2.9 examples/sec; 0.344 sec/batch)
  
  ```

  

- Step 2, train decision net

  ```bash
  python3 -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --size_height=1408 --size_width=512 --with_seg_net=False --with_decision_net=True --storage_dir=output --dataset_dir=db --datasets=kolektorSDD-dilate=5 --name_prefix=decision-net_full-size_cross-entropy --pretrained_main_folder=output/segdec_train/kolektorSDD-dilate\=5/full-size_cross-entropy/
  ```

  the results

  ```bash
  2019-09-12 10:14:01.929985: step 6567, loss = 0.00070 (8.1 examples/sec; 0.124 sec/batch)
  2019-09-12 10:14:03.291748: step 6578, loss = 0.00007 (8.0 examples/sec; 0.124 sec/batch)
  2019-09-12 10:14:04.659136: step 6589, loss = 0.00013 (8.0 examples/sec; 0.125 sec/batch)
  2019-09-12 10:14:07.008806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
  2019-09-12 10:14:07.008884: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
  2019-09-12 10:14:07.008903: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
  2019-09-12 10:14:07.008918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
  2019-09-12 10:14:07.009154: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7455 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
  Successfully loaded model from output/segdec_train/kolektorSDD-dilate=5/decision-net_full-size_cross-entropy/fold_2/model.ckpt-6599 at step=6599.
  2019-09-12 10:14:07.117994: starting evaluation on ().
  2019-09-12 10:14:15.513394: [20 batches out of 128] (2.4 examples/sec; 0.420sec/batch)
  2019-09-12 10:14:23.738696: [40 batches out of 128] (2.4 examples/sec; 0.411sec/batch)
  2019-09-12 10:14:32.296752: [60 batches out of 128] (2.3 examples/sec; 0.428sec/batch)
  2019-09-12 10:14:40.837171: [80 batches out of 128] (2.3 examples/sec; 0.427sec/batch)
  2019-09-12 10:14:49.509094: [100 batches out of 128] (2.3 examples/sec; 0.434sec/batch)
  2019-09-12 10:14:58.171020: [120 batches out of 128] (2.3 examples/sec; 0.433sec/batch)
  AUC=0.999442, and AP=0.996209, with best thr=0.011063 at f-measure=0.970 and FP=1, FN=0
  
  ```

  

- Step3, print evaluation metrics combined from all folds

  ```bash
  python3 -u segdec_print_eval.py output/segdec_eval/kolektorSDD-dilate\=5/decision-net_full-size_cross-entropy/
  ```

  results

  ```bash
  AP: 0.982, FP/FN: 1/2, FP@FN=0: 38
  ```

  





## How to convert the images/annotation to TFRecord files & check the conversion is OK.

the original source code need to modify & generate the dataset.

see *input_data/input_data_build_image_data_with_mask.py* for reference.



### train the generated data & compare results with results of original data

then train using the generated data

```bash
# results
2019-09-19 14:03:44.130679: step 6193, loss = 0.00972 (3.0 examples/sec; 0.336 sec/batch)
2019-09-19 14:03:47.859391: step 6204, loss = 0.00195 (3.0 examples/sec; 0.339 sec/batch)
2019-09-19 14:03:51.587183: step 6215, loss = 0.01393 (3.0 examples/sec; 0.339 sec/batch)
2019-09-19 14:03:55.326956: step 6226, loss = 0.00189 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:03:59.054307: step 6237, loss = 0.00976 (3.0 examples/sec; 0.336 sec/batch)
2019-09-19 14:04:02.779567: step 6248, loss = 0.00188 (2.9 examples/sec; 0.339 sec/batch)
2019-09-19 14:04:06.510376: step 6259, loss = 0.00878 (2.9 examples/sec; 0.341 sec/batch)
2019-09-19 14:04:11.548586: step 6270, loss = 0.00187 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:04:15.285691: step 6281, loss = 0.01080 (2.9 examples/sec; 0.342 sec/batch)
2019-09-19 14:04:19.023301: step 6292, loss = 0.00187 (3.0 examples/sec; 0.339 sec/batch)
2019-09-19 14:04:22.761992: step 6303, loss = 0.01023 (3.0 examples/sec; 0.338 sec/batch)
2019-09-19 14:04:26.491448: step 6314, loss = 0.00193 (3.0 examples/sec; 0.336 sec/batch)
2019-09-19 14:04:30.228721: step 6325, loss = 0.01102 (3.0 examples/sec; 0.337 sec/batch)
2019-09-19 14:04:33.965265: step 6336, loss = 0.00186 (3.0 examples/sec; 0.337 sec/batch)
2019-09-19 14:04:37.701051: step 6347, loss = 0.00886 (3.0 examples/sec; 0.338 sec/batch)
2019-09-19 14:04:41.432109: step 6358, loss = 0.00195 (3.0 examples/sec; 0.336 sec/batch)
2019-09-19 14:04:45.170063: step 6369, loss = 0.01054 (2.9 examples/sec; 0.339 sec/batch)
2019-09-19 14:04:50.294438: step 6380, loss = 0.00185 (2.9 examples/sec; 0.342 sec/batch)
2019-09-19 14:04:54.042942: step 6391, loss = 0.00921 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:04:57.798334: step 6402, loss = 0.00189 (2.9 examples/sec; 0.339 sec/batch)
2019-09-19 14:05:01.546362: step 6413, loss = 0.00869 (2.9 examples/sec; 0.339 sec/batch)
2019-09-19 14:05:05.278984: step 6424, loss = 0.00186 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:05:09.022486: step 6435, loss = 0.00520 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:05:12.761802: step 6446, loss = 0.00187 (2.9 examples/sec; 0.341 sec/batch)
2019-09-19 14:05:16.515922: step 6457, loss = 0.00908 (2.9 examples/sec; 0.342 sec/batch)
2019-09-19 14:05:20.265058: step 6468, loss = 0.00184 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:05:24.007003: step 6479, loss = 0.01043 (2.9 examples/sec; 0.341 sec/batch)
2019-09-19 14:05:29.122044: step 6490, loss = 0.00186 (3.0 examples/sec; 0.337 sec/batch)
2019-09-19 14:05:32.865167: step 6501, loss = 0.01095 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:05:36.605265: step 6512, loss = 0.00184 (2.9 examples/sec; 0.342 sec/batch)
2019-09-19 14:05:40.348156: step 6523, loss = 0.00418 (2.9 examples/sec; 0.339 sec/batch)
2019-09-19 14:05:44.096124: step 6534, loss = 0.00184 (2.9 examples/sec; 0.341 sec/batch)
2019-09-19 14:05:47.841580: step 6545, loss = 0.00961 (2.9 examples/sec; 0.341 sec/batch)
2019-09-19 14:05:51.586821: step 6556, loss = 0.00184 (3.0 examples/sec; 0.338 sec/batch)
2019-09-19 14:05:55.340773: step 6567, loss = 0.01156 (3.0 examples/sec; 0.339 sec/batch)
2019-09-19 14:05:59.085847: step 6578, loss = 0.00191 (2.9 examples/sec; 0.340 sec/batch)
2019-09-19 14:06:02.830728: step 6589, loss = 0.00861 (2.9 examples/sec; 0.339 sec/batch)
```

continue to the step2

```bash
 python3 -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --size_height=1408 --size_width=512 --with_    seg_net=False --with_decision_net=True --storage_dir=output --dataset_dir=db --datasets=output --name_prefix=decision-net_full-size_cross-entropy --pret    rained_main_folder=output/segdec_train/output/full-size_cross-entropy/
```

and the results is 

```bash
2019-09-19 14:57:35.154978: step 6556, loss = 0.00002 (8.1 examples/sec; 0.123 sec/batch)
2019-09-19 14:57:36.523085: step 6567, loss = 0.00005 (7.8 examples/sec; 0.128 sec/batch)
2019-09-19 14:57:37.882461: step 6578, loss = 0.00001 (8.2 examples/sec; 0.122 sec/batch)
2019-09-19 14:57:39.240746: step 6589, loss = 0.00011 (8.2 examples/sec; 0.122 sec/batch)
2019-09-19 14:57:41.261680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-09-19 14:57:41.261725: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-09-19 14:57:41.261731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-09-19 14:57:41.261736: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-09-19 14:57:41.261846: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6965 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
Successfully loaded model from output/segdec_train/output/decision-net_full-size_cross-entropy/fold_2/model.ckpt-6599 at step=6599.
2019-09-19 14:57:41.354713: starting evaluation on ().
2019-09-19 14:57:49.932947: [20 batches out of 128] (2.3 examples/sec; 0.429sec/batch)
2019-09-19 14:57:58.471700: [40 batches out of 128] (2.3 examples/sec; 0.427sec/batch)
2019-09-19 14:58:06.805957: [60 batches out of 128] (2.4 examples/sec; 0.417sec/batch)
2019-09-19 14:58:15.429744: [80 batches out of 128] (2.3 examples/sec; 0.431sec/batch)
2019-09-19 14:58:24.051719: [100 batches out of 128] (2.3 examples/sec; 0.431sec/batch)
2019-09-19 14:58:32.734229: [120 batches out of 128] (2.3 examples/sec; 0.434sec/batch)
AUC=0.996652, and AP=0.980711, with best thr=0.665497 at f-measure=0.938 and FP=1, FN=1

```

 The results are almost the same as the predefined data.





- ref:

1. http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

2. https://www.jianshu.com/p/27e4dc070761, follow this instruction & use the official tf_coco_records.py to generate the tfrecords file.

3. https://stackoverflow.com/questions/31064981/python3-error-initial-value-must-be-str-or-none, from here, I found the solution, the original code has some old code.

   

### generate the new custom data

 split the new data into three folder & generate the custom data for tensorflow, then training with the custom data

```bash
python3 segdec_train.py  --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --size_height=600 --size_width=8192 --with_seg_net=True --with_decision_net=False --storage_dir=silverplot --dataset_dir=db --datasets=SilverPlot --name_prefix=full-size_cross-entropy
```



When training , at first stage, the loss seems ok.

but it will ran out of memory during folder_1 training (the 2nd stage of the seg training)

```bash
2019-09-20 17:21:20.793348: step 2167, loss = 0.00931 (0.4 examples/sec; 2.357 sec/batch)
2019-09-20 17:21:46.743566: step 2178, loss = 0.00689 (0.4 examples/sec; 2.352 sec/batch)
2019-09-20 17:22:12.717685: step 2189, loss = 0.00748 (0.4 examples/sec; 2.363 sec/batch)
2019-09-20 17:22:46.713934: W tensorflow/core/common_runtime/bfc_allocator.cc:267] Allocator (GPU_0_bfc) ran out of memory trying to allocate 300.00MiB.  Current allocation summary follows.
2019-09-20 17:22:46.714004: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (256): 	Total Chunks: 411, Chunks in use: 405. 102.8KiB allocated for chunks. 101.2KiB in use in bin. 44.8KiB client-requested in use in bin.
2019-09-20 17:22:46.714022: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (512): 	Total Chunks: 4, Chunks in use: 0. 2.0KiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2019-09-20 17:22:46.714036: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (1024): 	Total Chunks: 2, Chunks in use: 2. 2.5KiB allocated for chunks. 2.5KiB in use in bin. 2.0KiB client-requested in use in bin.
2019-09-20 17:22:46.714049: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (2048): 	Total Chunks: 4, Chunks in use: 4. 13.0KiB allocated for chunks. 13.0KiB in use in bin. 12.5KiB client-requested in use in bin.
2019-09-20 17:22:46.714061: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (4096): 	Total Chunks: 24, Chunks in use: 24. 96.0KiB allocated for chunks. 96.0KiB in use in bin. 96.0KiB client-requested in use in bin.
2019-09-20 17:22:46.714074: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (8192): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2019-09-20 17:22:46.714088: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (16384): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2019-09-20 17:22:46.714101: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (32768): 	Total Chunks: 1, Chunks in use: 0. 59.0KiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2019-09-20 17:22:46.714115: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (65536): 	Total Chunks: 4, Chunks in use: 4. 400.0KiB allocated for chunks. 400.0KiB in use in bin. 400.0KiB client-requested in use in bin.
2019-09-20 17:22:46.714128: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (131072): 	Total Chunks: 4, Chunks in use: 4. 800.0KiB allocated for chunks. 800.0KiB in use in bin. 800.0KiB client-requested in use in bin.
2019-09-20 17:22:46.714143: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (262144): 	Total Chunks: 28, Chunks in use: 28. 10.68MiB allocated for chunks. 10.68MiB in use in bin. 10.45MiB client-requested in use in bin.
2019-09-20 17:22:46.714157: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (524288): 	Total Chunks: 1, Chunks in use: 1. 655.5KiB allocated for chunks. 655.5KiB in use in bin. 400.0KiB client-requested in use in bin.
2019-09-20 17:22:46.714167: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (1048576): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2019-09-20 17:22:46.714180: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (2097152): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2019-09-20 17:22:46.714192: I tensorflow/core/common_runtime/bfc_allocator.cc:597] Bin (4194304): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.

```



It seems the batch size is too large. So I try to use a smaller training-size of batch-size , and retry again.

```bash

```

Now, the train loss is loss:

```bash
2019-09-20 22:17:03.061670: step 6358, loss = 0.00418 (3.0 examples/sec; 0.330 sec/batch)
2019-09-20 22:17:07.846348: step 6369, loss = 0.00758 (3.0 examples/sec; 0.333 sec/batch)
2019-09-20 22:17:24.717371: step 6380, loss = 0.00430 (3.0 examples/sec; 0.328 sec/batch)
2019-09-20 22:17:32.781408: step 6391, loss = 0.00998 (1.6 examples/sec; 0.616 sec/batch)
2019-09-20 22:17:40.141161: step 6402, loss = 0.00525 (3.0 examples/sec; 0.334 sec/batch)
2019-09-20 22:17:47.188868: step 6413, loss = 0.01010 (0.4 examples/sec; 2.620 sec/batch)
2019-09-20 22:17:51.949853: step 6424, loss = 0.00445 (3.0 examples/sec; 0.329 sec/batch)
2019-09-20 22:18:05.676934: step 6435, loss = 0.05128 (0.6 examples/sec; 1.733 sec/batch)
2019-09-20 22:18:16.036154: step 6446, loss = 0.00505 (3.0 examples/sec; 0.333 sec/batch)
2019-09-20 22:18:23.344258: step 6457, loss = 0.00756 (3.0 examples/sec; 0.331 sec/batch)
2019-09-20 22:18:30.434938: step 6468, loss = 0.00544 (3.0 examples/sec; 0.330 sec/batch)
2019-09-20 22:18:35.203630: step 6479, loss = 0.00689 (1.8 examples/sec; 0.562 sec/batch)
2019-09-20 22:18:50.218606: step 6490, loss = 0.00466 (3.0 examples/sec; 0.332 sec/batch)
2019-09-20 22:19:00.085545: step 6501, loss = 0.49548 (0.6 examples/sec; 1.740 sec/batch)
2019-09-20 22:19:07.696817: step 6512, loss = 0.00489 (3.0 examples/sec; 0.332 sec/batch)
2019-09-20 22:19:12.468754: step 6523, loss = 0.02237 (3.0 examples/sec; 0.328 sec/batch)
2019-09-20 22:19:19.542478: step 6534, loss = 0.00490 (3.0 examples/sec; 0.331 sec/batch)
2019-09-20 22:19:31.881178: step 6545, loss = 0.01098 (3.0 examples/sec; 0.332 sec/batch)
2019-09-20 22:19:43.354295: step 6556, loss = 0.00487 (3.0 examples/sec; 0.332 sec/batch)
2019-09-20 22:19:51.012557: step 6567, loss = 0.01031 (3.0 examples/sec; 0.331 sec/batch)
2019-09-20 22:19:55.812499: step 6578, loss = 0.00472 (3.0 examples/sec; 0.333 sec/batch)
2019-09-20 22:20:02.645117: step 6589, loss = 0.00572 (0.8 examples/sec; 1.181 sec/batch)

```

Then train with the decision net

```bash
python3 -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --with_seg_net=False --with_decision_net=True --storage_dir=silverplot --dataset_dir=db --datasets=SilverPlot --name_prefix=decision-net_full-size_cross-entropy --pretrained_main_folder=silverplot/segdec_train/SilverPlot/full-size_cross-entropy/
```

and the results are

- ```bash
  AUC=0.765271, and AP=0.970598, with best thr=0.002465 at f-measure=0.972 and FP=11, FN=0
  ```

- ```bash
  AUC=0.497973, and AP=0.926816, with best thr=0.001380 at f-measure=0.959 and FP=16, FN=0
  ```

- 3rd cross-fold

```bash
2019-09-21 03:15:56.251626: step 6556, loss = 0.00036 (8.3 examples/sec; 0.120 sec/batch)
2019-09-21 03:16:01.671897: step 6567, loss = 0.00001 (2.5 examples/sec; 0.395 sec/batch)
2019-09-21 03:16:09.617940: step 6578, loss = 0.00625 (8.4 examples/sec; 0.120 sec/batch)
2019-09-21 03:16:21.162833: step 6589, loss = 0.00000 (8.2 examples/sec; 0.121 sec/batch)
2019-09-21 03:16:46.719843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-09-21 03:16:46.719893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-09-21 03:16:46.719905: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-09-21 03:16:46.719914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-09-21 03:16:46.720098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7394 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
Successfully loaded model from silverplot/segdec_train/SilverPlot/decision-net_full-size_cross-entropy/fold_2/model.ckpt-6599 at step=6599.
2019-09-21 03:16:46.862849: starting evaluation on ().
2019-09-21 03:16:57.512891: [20 batches out of 203] (1.9 examples/sec; 0.533sec/batch)
2019-09-21 03:17:08.347202: [40 batches out of 203] (1.8 examples/sec; 0.542sec/batch)
2019-09-21 03:17:18.840198: [60 batches out of 203] (1.9 examples/sec; 0.525sec/batch)
2019-09-21 03:17:29.763205: [80 batches out of 203] (1.8 examples/sec; 0.546sec/batch)
2019-09-21 03:17:40.742820: [100 batches out of 203] (1.8 examples/sec; 0.549sec/batch)
2019-09-21 03:17:51.515734: [120 batches out of 203] (1.9 examples/sec; 0.539sec/batch)
2019-09-21 03:18:02.526013: [140 batches out of 203] (1.8 examples/sec; 0.551sec/batch)
2019-09-21 03:18:13.248233: [160 batches out of 203] (1.9 examples/sec; 0.536sec/batch)
2019-09-21 03:18:24.388921: [180 batches out of 203] (1.8 examples/sec; 0.557sec/batch)
2019-09-21 03:18:35.254661: [200 batches out of 203] (1.8 examples/sec; 0.543sec/batch)
AUC=0.743517, and AP=0.962517, with best thr=0.000824 at f-measure=0.956 and FP=17, FN=0

```







## Use Type4 data for checking

### step1. generate masks using generate.py

### step2. split the masks&image using split_files.py

### step3. mkdir & generate the tfrecords 

​	using script *input_data/input_data_build_image_data_with_mask.py*, this may take a long time.



### step4. segdec training

```bash
python3 segdec_train.py  --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --with_seg_net=True --with_decision_net=False --storage_dir=silverplot --dataset_dir=db --datasets=SilverPlot --name_prefix=full-size_cross-entropy

```

results for the 1st stage

```bash
2019-09-24 10:45:10.682924: step 6424, loss = 0.00621 (2.9 examples/sec; 0.343 sec/batch)
2019-09-24 10:45:14.447287: step 6435, loss = 0.02465 (2.9 examples/sec; 0.339 sec/batch)
2019-09-24 10:45:18.190844: step 6446, loss = 0.00594 (2.9 examples/sec; 0.340 sec/batch)
2019-09-24 10:45:21.933582: step 6457, loss = 0.01901 (2.9 examples/sec; 0.341 sec/batch)
2019-09-24 10:45:25.676488: step 6468, loss = 0.00491 (3.0 examples/sec; 0.338 sec/batch)
2019-09-24 10:45:29.425709: step 6479, loss = 0.01842 (2.9 examples/sec; 0.340 sec/batch)
2019-09-24 10:45:34.451769: step 6490, loss = 0.00595 (2.9 examples/sec; 0.341 sec/batch)
2019-09-24 10:45:38.214521: step 6501, loss = 0.01154 (2.9 examples/sec; 0.343 sec/batch)
2019-09-24 10:45:41.973229: step 6512, loss = 0.00441 (2.9 examples/sec; 0.345 sec/batch)
2019-09-24 10:45:45.723450: step 6523, loss = 0.01874 (2.9 examples/sec; 0.340 sec/batch)
2019-09-24 10:45:49.460880: step 6534, loss = 0.00735 (3.0 examples/sec; 0.339 sec/batch)
2019-09-24 10:45:53.197467: step 6545, loss = 0.01004 (3.0 examples/sec; 0.338 sec/batch)
2019-09-24 10:45:56.943688: step 6556, loss = 0.00366 (2.9 examples/sec; 0.339 sec/batch)
2019-09-24 10:46:00.701106: step 6567, loss = 0.01446 (2.9 examples/sec; 0.340 sec/batch)
2019-09-24 10:46:04.454207: step 6578, loss = 0.00456 (2.9 examples/sec; 0.340 sec/batch)
2019-09-24 10:46:08.213346: step 6589, loss = 0.01210 (2.9 examples/sec; 0.342 sec/batch)

```

The loss is not good enough , may be this is caused by the input image_size settings.



2nd stage. training decision net with seg-net weights

```bash
python3 -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train --seg_net_type=ENTROPY --with_seg_net=False     --with_decision_net=True --storage_dir=silverplot --dataset_dir=db --datasets=SilverPlot --name_prefix=decision-net_full-size_cr    oss-entropy --pretrained_main_folder=silverplot/segdec_train/SilverPlot/full-size_cross-entropy/
```

results:

- fold0:

  ```bash
  2019-09-24 11:10:00.109815: step 6523, loss = 0.00857 (8.1 examples/sec; 0.124 sec/batch)
  2019-09-24 11:10:01.461508: step 6534, loss = 0.00683 (8.2 examples/sec; 0.123 sec/batch)
  2019-09-24 11:10:02.808911: step 6545, loss = 0.54139 (8.3 examples/sec; 0.121 sec/batch)
  2019-09-24 11:10:04.159796: step 6556, loss = 0.19290 (8.1 examples/sec; 0.123 sec/batch)
  2019-09-24 11:10:05.511090: step 6567, loss = 0.00491 (8.2 examples/sec; 0.122 sec/batch)
  2019-09-24 11:10:06.864034: step 6578, loss = 0.78183 (7.9 examples/sec; 0.126 sec/batch)
  2019-09-24 11:10:08.221674: step 6589, loss = 0.21312 (8.2 examples/sec; 0.122 sec/batch)
  2019-09-24 11:10:23.084048: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
  2019-09-24 11:10:23.084117: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
  2019-09-24 11:10:23.084133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
  2019-09-24 11:10:23.084145: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
  2019-09-24 11:10:23.084346: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7394 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
  Successfully loaded model from silverplot/segdec_train/SilverPlot/decision-net_full-size_cross-entropy/fold_0/model.ckpt-6599 at step=6599.
  2019-09-24 11:10:23.196143: starting evaluation on ().
  
  
  2019-09-24 11:15:38.431020: [680 batches out of 885] (2.1 examples/sec; 0.479sec/batch)
  2019-09-24 11:15:48.144032: [700 batches out of 885] (2.1 examples/sec; 0.486sec/batch)
  2019-09-24 11:15:57.839466: [720 batches out of 885] (2.1 examples/sec; 0.485sec/batch)
  2019-09-24 11:16:07.512764: [740 batches out of 885] (2.1 examples/sec; 0.484sec/batch)
  2019-09-24 11:16:17.175465: [760 batches out of 885] (2.1 examples/sec; 0.483sec/batch)
  2019-09-24 11:16:27.097573: [780 batches out of 885] (2.0 examples/sec; 0.496sec/batch)
  2019-09-24 11:16:36.762373: [800 batches out of 885] (2.1 examples/sec; 0.483sec/batch)
  2019-09-24 11:16:46.481376: [820 batches out of 885] (2.1 examples/sec; 0.486sec/batch)
  2019-09-24 11:16:56.340340: [840 batches out of 885] (2.0 examples/sec; 0.493sec/batch)
  2019-09-24 11:17:06.137473: [860 batches out of 885] (2.0 examples/sec; 0.490sec/batch)
  2019-09-24 11:17:16.099610: [880 batches out of 885] (2.0 examples/sec; 0.498sec/batch)
  AUC=0.892527, and AP=0.955233, with best thr=0.264638 at f-measure=0.909 and FP=82, FN=42
  
  ```

  

- fold1:

  ```bash
  2019-09-24 11:32:19.696902: step 6523, loss = 0.06678 (8.3 examples/sec; 0.121 sec/batch)
  2019-09-24 11:32:21.043178: step 6534, loss = 0.26186 (8.2 examples/sec; 0.122 sec/batch)
  2019-09-24 11:32:22.397068: step 6545, loss = 0.14080 (8.2 examples/sec; 0.123 sec/batch)
  2019-09-24 11:32:23.748862: step 6556, loss = 0.20673 (8.1 examples/sec; 0.123 sec/batch)
  2019-09-24 11:32:25.089737: step 6567, loss = 1.12496 (8.1 examples/sec; 0.123 sec/batch)
  2019-09-24 11:32:26.438573: step 6578, loss = 0.07576 (8.1 examples/sec; 0.123 sec/batch)
  2019-09-24 11:32:27.787450: step 6589, loss = 0.88555 (8.2 examples/sec; 0.122 sec/batch)
  2019-09-24 11:32:42.814154: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
  2019-09-24 11:32:42.814232: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
  2019-09-24 11:32:42.814252: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
  2019-09-24 11:32:42.814266: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
  2019-09-24 11:32:42.814522: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7394 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
  Successfully loaded model from silverplot/segdec_train/SilverPlot/decision-net_full-size_cross-entropy/fold_1/model.ckpt-6599 at step=6599.
  2019-09-24 11:32:43.102193: starting evaluation on ().
  
  2019-09-24 11:38:57.955913: [720 batches out of 887] (1.8 examples/sec; 0.555sec/batch)
  2019-09-24 11:39:08.709607: [740 batches out of 887] (1.9 examples/sec; 0.538sec/batch)
  2019-09-24 11:39:19.577183: [760 batches out of 887] (1.8 examples/sec; 0.543sec/batch)
  2019-09-24 11:39:30.788740: [780 batches out of 887] (1.8 examples/sec; 0.561sec/batch)
  2019-09-24 11:39:42.000691: [800 batches out of 887] (1.8 examples/sec; 0.561sec/batch)
  2019-09-24 11:39:52.891982: [820 batches out of 887] (1.8 examples/sec; 0.545sec/batch)
  2019-09-24 11:40:04.172092: [840 batches out of 887] (1.8 examples/sec; 0.564sec/batch)
  2019-09-24 11:40:15.124283: [860 batches out of 887] (1.8 examples/sec; 0.548sec/batch)
  2019-09-24 11:40:26.357907: [880 batches out of 887] (1.8 examples/sec; 0.562sec/batch)
  AUC=0.899893, and AP=0.966910, with best thr=0.318705 at f-measure=0.908 and FP=65, FN=61
  
  ```

  

- fold2:

```bash
2019-09-24 11:55:34.091092: step 6512, loss = 1.13906 (8.2 examples/sec; 0.122 sec/batch)
2019-09-24 11:55:35.440408: step 6523, loss = 0.08642 (8.4 examples/sec; 0.119 sec/batch)
2019-09-24 11:55:36.787003: step 6534, loss = 0.05353 (8.2 examples/sec; 0.122 sec/batch)
2019-09-24 11:55:38.135424: step 6545, loss = 0.29820 (8.2 examples/sec; 0.122 sec/batch)
2019-09-24 11:55:39.484171: step 6556, loss = 1.50303 (8.2 examples/sec; 0.122 sec/batch)
2019-09-24 11:55:40.838369: step 6567, loss = 0.52315 (8.2 examples/sec; 0.122 sec/batch)
2019-09-24 11:55:42.190112: step 6578, loss = 0.04155 (8.2 examples/sec; 0.122 sec/batch)
2019-09-24 11:55:43.543513: step 6589, loss = 0.04656 (8.1 examples/sec; 0.123 sec/batch)
2019-09-24 11:56:00.499388: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-09-24 11:56:00.499457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-09-24 11:56:00.499473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-09-24 11:56:00.499485: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-09-24 11:56:00.499685: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7394 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
Successfully loaded model from silverplot/segdec_train/SilverPlot/decision-net_full-size_cross-entropy/fold_2/model.ckpt-6599 at step=6599.
2019-09-24 12:03:47.933163: [780 batches out of 887] (1.5 examples/sec; 0.660sec/batch)
2019-09-24 12:04:00.449859: [800 batches out of 887] (1.6 examples/sec; 0.626sec/batch)
2019-09-24 12:04:13.087945: [820 batches out of 887] (1.6 examples/sec; 0.632sec/batch)
2019-09-24 12:04:26.047454: [840 batches out of 887] (1.5 examples/sec; 0.648sec/batch)
2019-09-24 12:04:38.823757: [860 batches out of 887] (1.6 examples/sec; 0.639sec/batch)
2019-09-24 12:04:51.867065: [880 batches out of 887] (1.5 examples/sec; 0.652sec/batch)
AUC=0.892026, and AP=0.964687, with best thr=0.158062 at f-measure=0.916 and FP=91, FN=31
```



summarizes it as table

|         | fold0,1/fold2                                                | fold0,2/fold1                                                | fold1,2/fold0                                                |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| results | AUC=0.892527, and AP=0.955233, with best thr=0.264638 at f-measure=0.909 and FP=82, FN=42 | AUC=0.899893, and AP=0.966910, with best thr=0.318705 at f-measure=0.908 and FP=65, FN=61 | AUC=0.892026, and AP=0.964687, with best thr=0.158062 at f-measure=0.916 and FP=91, FN=31 |
| #FOLD   | #fold_0=888                                                  | #fold1=887                                                   | #fold2=887                                                   |

FP, FN seems too high.



### Step5. Re-train using the actual input image size.

cont. with step3. seg-net training changed to 

```bash
2019-09-24 16:59:34.979166: step 6215, loss = 0.00863 (1.4 examples/sec; 0.731 sec/batch)
2019-09-24 16:59:43.161040: step 6226, loss = 0.00680 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 16:59:51.338709: step 6237, loss = 0.00909 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 16:59:59.516066: step 6248, loss = 0.00789 (1.4 examples/sec; 0.735 sec/batch)
2019-09-24 17:00:07.695445: step 6259, loss = 0.01025 (1.4 examples/sec; 0.740 sec/batch)
2019-09-24 17:00:18.215114: step 6270, loss = 0.00567 (1.3 examples/sec; 0.744 sec/batch)
2019-09-24 17:00:26.388678: step 6281, loss = 0.01537 (1.4 examples/sec; 0.737 sec/batch)
2019-09-24 17:00:34.575218: step 6292, loss = 0.00545 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 17:00:42.748067: step 6303, loss = 0.07591 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 17:00:50.912483: step 6314, loss = 0.00535 (1.4 examples/sec; 0.732 sec/batch)
2019-09-24 17:00:59.101637: step 6325, loss = 0.02145 (1.4 examples/sec; 0.737 sec/batch)
2019-09-24 17:01:07.290067: step 6336, loss = 0.00704 (1.3 examples/sec; 0.745 sec/batch)
2019-09-24 17:01:15.470192: step 6347, loss = 0.02304 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 17:01:23.651501: step 6358, loss = 0.00671 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:01:31.828894: step 6369, loss = 0.00862 (1.3 examples/sec; 0.742 sec/batch)
2019-09-24 17:01:42.484907: step 6380, loss = 0.00512 (1.4 examples/sec; 0.738 sec/batch)
2019-09-24 17:01:50.662214: step 6391, loss = 0.01438 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:01:58.838737: step 6402, loss = 0.00454 (1.4 examples/sec; 0.740 sec/batch)
2019-09-24 17:02:07.020667: step 6413, loss = 0.02485 (1.4 examples/sec; 0.740 sec/batch)
2019-09-24 17:02:15.188977: step 6424, loss = 0.00778 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:02:23.357936: step 6435, loss = 0.02152 (1.4 examples/sec; 0.732 sec/batch)
2019-09-24 17:02:31.545624: step 6446, loss = 0.00738 (1.3 examples/sec; 0.742 sec/batch)
2019-09-24 17:02:39.728594: step 6457, loss = 0.05156 (1.3 examples/sec; 0.742 sec/batch)
2019-09-24 17:02:47.927659: step 6468, loss = 0.00642 (1.3 examples/sec; 0.744 sec/batch)
2019-09-24 17:02:56.106755: step 6479, loss = 0.01685 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 17:03:06.692790: step 6490, loss = 0.00612 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:03:14.866296: step 6501, loss = 0.02035 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:03:23.048542: step 6512, loss = 0.00734 (1.3 examples/sec; 0.745 sec/batch)
2019-09-24 17:03:31.219012: step 6523, loss = 0.02908 (1.3 examples/sec; 0.741 sec/batch)
2019-09-24 17:03:39.407586: step 6534, loss = 0.00587 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:03:47.585298: step 6545, loss = 0.01406 (1.4 examples/sec; 0.739 sec/batch)
2019-09-24 17:03:55.777206: step 6556, loss = 0.00747 (1.4 examples/sec; 0.741 sec/batch)
2019-09-24 17:04:03.964764: step 6567, loss = 0.02049 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:04:12.159449: step 6578, loss = 0.01051 (1.3 examples/sec; 0.743 sec/batch)
2019-09-24 17:04:20.334215: step 6589, loss = 0.01630 (1.4 examples/sec; 0.740 sec/batch)

```

It looks like the previous 1st stage results.

continue with 2nd stage & get the results

```bash
2019-09-24 17:51:05.620153: [720 batches out of 885] (1.8 examples/sec; 0.555sec/batch)
2019-09-24 17:51:16.769493: [740 batches out of 885] (1.8 examples/sec; 0.557sec/batch)
2019-09-24 17:51:27.905009: [760 batches out of 885] (1.8 examples/sec; 0.557sec/batch)
2019-09-24 17:51:38.946124: [780 batches out of 885] (1.8 examples/sec; 0.552sec/batch)
2019-09-24 17:51:50.124697: [800 batches out of 885] (1.8 examples/sec; 0.559sec/batch)
2019-09-24 17:52:01.292475: [820 batches out of 885] (1.8 examples/sec; 0.558sec/batch)
2019-09-24 17:52:12.650422: [840 batches out of 885] (1.8 examples/sec; 0.568sec/batch)
2019-09-24 17:52:23.837238: [860 batches out of 885] (1.8 examples/sec; 0.559sec/batch)
2019-09-24 17:52:35.041477: [880 batches out of 885] (1.8 examples/sec; 0.560sec/batch)
AUC=0.896811, and AP=0.954784, with best thr=0.168253 at f-measure=0.916 and FP=88, FN=29


2019-09-24 18:31:01.206896: [640 batches out of 887] (1.6 examples/sec; 0.612sec/batch)
2019-09-24 18:31:13.476377: [660 batches out of 887] (1.6 examples/sec; 0.613sec/batch)
2019-09-24 18:31:25.487575: [680 batches out of 887] (1.7 examples/sec; 0.601sec/batch)
2019-09-24 18:31:37.859529: [700 batches out of 887] (1.6 examples/sec; 0.619sec/batch)
2019-09-24 18:31:50.016953: [720 batches out of 887] (1.6 examples/sec; 0.608sec/batch)
2019-09-24 18:32:02.547700: [740 batches out of 887] (1.6 examples/sec; 0.627sec/batch)
2019-09-24 18:32:14.617828: [760 batches out of 887] (1.7 examples/sec; 0.604sec/batch)
2019-09-24 18:32:26.956557: [780 batches out of 887] (1.6 examples/sec; 0.617sec/batch)
2019-09-24 18:32:39.419854: [800 batches out of 887] (1.6 examples/sec; 0.623sec/batch)
2019-09-24 18:32:51.649494: [820 batches out of 887] (1.6 examples/sec; 0.611sec/batch)
2019-09-24 18:33:04.138584: [840 batches out of 887] (1.6 examples/sec; 0.624sec/batch)
2019-09-24 18:33:16.463995: [860 batches out of 887] (1.6 examples/sec; 0.616sec/batch)
2019-09-24 18:33:29.046116: [880 batches out of 887] (1.6 examples/sec; 0.629sec/batch)
AUC=0.883435, and AP=0.962646, with best thr=0.217666 at f-measure=0.900 and FP=101, FN=41


2019-09-24 19:10:38.720419: [480 batches out of 887] (1.5 examples/sec; 0.651sec/batch)
2019-09-24 19:10:52.136391: [500 batches out of 887] (1.5 examples/sec; 0.671sec/batch)
2019-09-24 19:11:05.166374: [520 batches out of 887] (1.5 examples/sec; 0.651sec/batch)
2019-09-24 19:11:18.277415: [540 batches out of 887] (1.5 examples/sec; 0.656sec/batch)
2019-09-24 19:11:31.810981: [560 batches out of 887] (1.5 examples/sec; 0.677sec/batch)
2019-09-24 19:11:44.921216: [580 batches out of 887] (1.5 examples/sec; 0.656sec/batch)
2019-09-24 19:11:58.404351: [600 batches out of 887] (1.5 examples/sec; 0.674sec/batch)
2019-09-24 19:12:11.613493: [620 batches out of 887] (1.5 examples/sec; 0.660sec/batch)
2019-09-24 19:12:25.170722: [640 batches out of 887] (1.5 examples/sec; 0.678sec/batch)
2019-09-24 19:12:38.343076: [660 batches out of 887] (1.5 examples/sec; 0.659sec/batch)
2019-09-24 19:12:51.600752: [680 batches out of 887] (1.5 examples/sec; 0.663sec/batch)
2019-09-24 19:13:05.268419: [700 batches out of 887] (1.5 examples/sec; 0.683sec/batch)
2019-09-24 19:13:18.552953: [720 batches out of 887] (1.5 examples/sec; 0.664sec/batch)
2019-09-24 19:13:32.219702: [740 batches out of 887] (1.5 examples/sec; 0.683sec/batch)
2019-09-24 19:13:45.608097: [760 batches out of 887] (1.5 examples/sec; 0.669sec/batch)
2019-09-24 19:13:59.455165: [780 batches out of 887] (1.4 examples/sec; 0.692sec/batch)
2019-09-24 19:14:12.832762: [800 batches out of 887] (1.5 examples/sec; 0.669sec/batch)
2019-09-24 19:14:26.267640: [820 batches out of 887] (1.5 examples/sec; 0.672sec/batch)
2019-09-24 19:14:40.050196: [840 batches out of 887] (1.5 examples/sec; 0.689sec/batch)
2019-09-24 19:14:53.561564: [860 batches out of 887] (1.5 examples/sec; 0.676sec/batch)
2019-09-24 19:15:07.463539: [880 batches out of 887] (1.4 examples/sec; 0.695sec/batch)
AUC=0.884384, and AP=0.959865, with best thr=0.188951 at f-measure=0.913 and FP=98, FN=29

```

summarizes the results as table

|         | fold0,1/fold2                                                | fold0,2/fold1                                                | fold1,2/fold0                                                |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| results | AUC=0.896811, and AP=0.954784, with best thr=0.168253 at f-measure=0.916 and FP=88, FN=29 | AUC=0.883435, and AP=0.962646, with best thr=0.217666 at f-measure=0.900 and FP=101, FN=41 | AUC=0.884384, and AP=0.959865, with best thr=0.188951 at f-measure=0.913 and FP=98, FN=29 |
| #FOLD   | #fold_0=888                                                  | #fold1=887                                                   | #fold2=887                                                   |





## Inspect the results

by reviewing the results(*pdfs), I found there some error in original annotated data. 

** TODO**, Need update the original annotation, and re-train again.

