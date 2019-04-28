import os, shutil

original_dataset_dir = "/media/mash-compute/eatsmart/DeepLearning/keras_revisited/data/data_master/train"
base_dir = "/media/mash-compute/eatsmart/DeepLearning/keras_revisited/data/data_processed"

def file_creator(dir_path):
    """
    Args:
        dir_path: path to the directory where the dir needs to be created
    Returns: None
    """
    try:
        os.mkdir(dir_path)
    except:
        print("{} dir already exists!".format(os.path.basename(dir_path)))

file_creator(base_dir)

train_dir = os.path.join(base_dir, 'train')
file_creator(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
file_creator(validation_dir)

test_dir = os.path.join(base_dir, 'test')
file_creator(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
file_creator(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
file_creator(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
file_creator(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
file_creator(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
file_creator(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
file_creator(test_dogs_dir)

# copy files to the created directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)