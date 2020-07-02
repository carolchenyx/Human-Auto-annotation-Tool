import json
import config


json_filename = config.json_filename

json_new = open(config.output_jsonname, "w")
with open(json_filename) as f:
    pop_data = json.load(f)
    pop_data_images = pop_data["images"]
    pop_data_annotations = pop_data["annotations"]
    for i in range(len(pop_data_images)):
        name = pop_data_images[i]['file_name']
        post_name = name.zfill(15)

	#remember to change the number according to the images
        post_name = config.imageandjson_first_number + post_name



        pop_data_images[i]['file_name'] = pop_data_images[i]['file_name'].replace(name,post_name)

    json_new.write(json.dumps(pop_data))
