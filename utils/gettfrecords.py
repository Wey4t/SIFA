import numpy as np
import os
import tensorflow as tf
import nibabel as nib

def npz2tfrecords():
    image_pth = './example data/image_01.npz'
    tfrecord_pth = './example data/image_01.tfrecords'

    npz_data = np.load(image_pth)
    data_vol_val = npz_data['arr_0']
    label_vol_val = npz_data['arr_1']
    dsize_dim0_val = data_vol_val.shape[0]
    dsize_dim1_val = data_vol_val.shape[1]
    dsize_dim2_val = data_vol_val.shape[2]
    lsize_dim0_val = label_vol_val.shape[0]
    lsize_dim1_val = label_vol_val.shape[1]
    lsize_dim2_val = label_vol_val.shape[2]

    writer = tf.python_io.TFRecordWriter(tfrecord_pth)

    feature = {'data_vol': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(data_vol_val.tostring())])),
        'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim0_val])),
        'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim1_val])),
        'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim2_val])),
        'lsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim0_val])),
        'lsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim1_val])),
        'lsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim2_val])),
        'label_vol': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label_vol_val.tostring())])), }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()

def nii2tfrecord(image_nii_path, label_nii_path, tfrecord_path):
    image_nii = nib.load(image_nii_path)
    label_nii = nib.load(label_nii_path)

    image_data = image_nii.get_fdata().astype(np.float32)
    label_data = label_nii.get_fdata().astype(np.float32)

    # Optionally pick a slice or reshape
    if image_data.ndim == 3:
        image_data = image_data[:, :, :3]  # first 3 slices for example
    if label_data.ndim == 3:
        label_data = label_data[:, :, :3]

    # You may want to select only the middle slice of the label
    label_data = label_data[:, :, 1:2]  # shape becomes [H, W, 1]

    dsize = image_data.shape
    lsize = label_data.shape

    feature = {
        'data_vol': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_data.tobytes()])),
        'label_vol': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[label_data.tobytes()])),
        'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize[0]])),
        'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize[1]])),
        'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize[2]])),
        'lsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize[0]])),
        'lsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize[1]])),
        'lsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize[2]])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    writer.write(example.SerializeToString())
    writer.close()


if __name__=='__main__':
    # npz2tfrecords()
    img = "FLARE22_Tr_0001_0000.nii.gz"
    label = "FLARE22_Tr_0001.nii.gz"
    nii2tfrecord(img, label, "Flare22.tfrecords")
