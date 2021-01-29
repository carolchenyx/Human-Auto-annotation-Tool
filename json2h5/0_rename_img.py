import os
import shutil
import config


img_folder = config.img_folder
dest_folder = config.dest_folder
os.makedirs(dest_folder, exist_ok=True)
img_names = [i for i in os.listdir(img_folder)]

for name in img_names:
    post_name = name.zfill(15)
    post_name = config.imageandjson_first_number + post_name
    print(post_name)
    shutil.copy(os.path.join(img_folder, name), os.path.join(dest_folder, post_name))
