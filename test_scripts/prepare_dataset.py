import pickle
import os.path as osp
import tensorflow as tf
import random
from convert_images_to_tfrecords import *

def _find_image_files(data_dir_list, data_ext, mask_pattern, ignore_non_positive_masks=False):
    """Build a list of all images files and labels in the data set.
        Args:
      data_dir_list: array of string, list of paths to the root directory of images.

        Assumes that the image data set resides in JPEG files located in
        the following directory structure.

          data_dir[0]/another-image.JPEG
          data_dir[1]/my-image.jpg

        with corresponding mask files in the same folders

          data_dir[0]/another-image_mask.png
          data_dir[1]/my-image_mask.png

      data_ext: string, extension of images
      mask_pattern: tuple of string, with mask_pattern[0] string replace pattern
        and mask_pattern[1] replace string, e.g., mask_pattern = ('.jpg', '_mask.png')
      ignore_non_positive_masks: boolean, ignores files that have only zero mask values

    Returns:
      filenames: list of strings; each string is a path to an image file.
      mask_filenames: list of strings; each string is a path to an image mask file.
    """


    filenames = []
    mask_filenames = []

    # Construct the list of JPEG files and labels.
    for data_dir in data_dir_list:
        print('Determining list of input files and labels from %s.' % data_dir)

        jpeg_file_path = '%s/*%s' % (data_dir, data_ext)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        # remove files that are actual labels !!
        matching_files = [f for f in matching_files if not f.endswith(mask_pattern[1])]
        print('matching_files:', matching_files)  # debug
        filenames.extend(matching_files)

        # Find corresponding mask files
        matching_files_masks = [filename.replace(mask_pattern[0], mask_pattern[1]) for filename in matching_files]

        mask_filenames.extend(matching_files_masks)

    if ignore_non_positive_masks:
        select = [np.any(Image.open(m)) for m in mask_filenames if os.path.exists(m)]
        filenames = [f for f, s in zip(filenames, select) if s]
        mask_filenames = [m for m, s in zip(mask_filenames, select) if s]

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    mask_filenames = [mask_filenames[i] for i in shuffled_index]

    print('Found %d image files across all folders.' % (len(filenames)))

    return filenames, mask_filenames


def main():
    name = "KolektorSDD-dilate5-mine"

    data_ext = '.jpg'
    mask_pattern = ('.jpg', '_label.bmp')

    with open('split.pyb', 'rb') as f:
        [train_split, test_split, all] = pickle.load(f)
    img_root = './db/original_dataset/'
    output_root = 'db/output'

    for i in range(len(train_split)):
        directory_list = []
        folder_name = "fold_%d" % i
        folder_root = osp.join(output_root, folder_name)
        output_tf_name = osp.join(folder_root, 'train.tfrecords')

        # for train-set
        for train_dir in train_split[i]:
            directory_list.append(osp.join(img_root, train_dir))
        filenames, mask_filenames = _find_image_files(directory_list, data_ext, mask_pattern)

        generator = TFRecordGenerator(output_tf_name)
        generator.convert(zip(filenames, mask_filenames))
        generator.closeConvert()
        with open(osp.join(folder_root,'train_ids.txt'),'w') as train_ids:
            for filename in filenames:
                train_ids.write(filename+"\n")
            train_ids.close()

        # for test records
        test_dir_list = []
        for test_dir in test_split[i]:
            test_dir_list.append(osp.join(img_root, test_dir))

        test_tf_name = osp.join(folder_root, 'test.tfrecords')
        generator = TFRecordGenerator(test_tf_name)
        filenames, mask_filenames = _find_image_files(test_dir_list, data_ext, mask_pattern)
        generator.convert(zip(filenames, mask_filenames))
        generator.closeConvert()
        with open(osp.join(folder_root, 'test_ids.txt'), 'w') as test_ids:
            for filename in filenames:
                test_ids.write(filename+"\n")
            test_ids.close()


if __name__ == "__main__":
    main()

