import random
import os
import shutil

src_folder = "Img2json/img/ceiling_pre_bad"
# dest_folder = src_folder + "+selected"
# os.makedirs(dest_folder, exist_ok=True)

src_ls = [os.path.join(src_folder, file) for file in os.listdir(src_folder)]

random_ls = random.sample(src_ls, 300)
for file in random_ls:
    os.remove(file)
