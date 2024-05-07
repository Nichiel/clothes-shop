import os
import yaml

config = yaml.safe_load(open('config.yaml'))

model_for_image_config = config['model_for_image']
model_for_image_name = config['model_for_image']['name']
model_for_image_embedding_size = config['model_for_image']['embedding_size']
model_for_image_label = config['model_for_image']['label']
model_for_text_config = config['model_for_text']
model_for_text_name = config['model_for_text']['name']
model_for_text_embedding_size = config['model_for_text']['embedding_size']
model_for_text_label = config['model_for_text']['label']

collection_name = config['vdb']['collection_name']

max_text_length = config['max_text_length']
