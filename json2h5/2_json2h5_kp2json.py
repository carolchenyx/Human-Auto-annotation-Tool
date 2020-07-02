import json
import numpy as np
import h5py
import config


def extract_info():
    is_first = 0

    for i in range(len(config.input_json)):

        with open(config.input_json[i], "r") as load_f:
            load_dict = json.load(load_f)
            anno = load_dict["images"]
            images_dict = load_dict["images"]
            if i == 0:
                bndboxes = np.array([])
                parts = np.array([])
                imgnames = np.array([])

            for i in range(len(images_dict)):
                if is_first == 0:
                    bndboxes = np.array(images_dict[i]["bbox"]).reshape(1, 1, -1)
                else:
                    bndboxes = np.concatenate(
                        (bndboxes, np.array(images_dict[i]["bbox"]).reshape(1, 1, -1))
                    )

                temp_name = np.fromstring(
                    images_dict[i]["image_name"], dtype=np.uint8
                )
                print(temp_name)
                for i in range(16 - len(temp_name)):
                    temp_name = np.append(temp_name, -1)
                print(temp_name)

                if is_first == 0:
                    imgnames = temp_name.reshape(1, 16)
                else:
                    imgnames = np.concatenate((imgnames, temp_name.reshape(1, 16)))

                keypoints = images_dict[i]["keypoints"]
                temp = np.array([])
                for k in range(0, len(keypoints), 2):
                    # if keypoints[k + 2] == 2:
                    if k == 0:
                        temp = np.array(keypoints[k : k + 2])
                    else:
                        temp = np.append(temp, keypoints[k : k + 2])

                for i in range(34 - len(temp)):
                    temp = np.append(temp, 0)
                temp = temp.reshape(1, 17, 2)
                if is_first == 0:
                    parts = temp
                else:
                    parts = np.concatenate((parts, temp))
                is_first += 1
            print(parts.shape)
            print(bndboxes.shape)
            print(imgnames.shape)

    f = h5py.File(config.output_h5name, "a")
    f.create_dataset("bndbox", data=bndboxes)
    f.create_dataset("imgname", data=imgnames)
    f.create_dataset("part", data=parts)


if __name__ == "__main__":
    extract_info()

