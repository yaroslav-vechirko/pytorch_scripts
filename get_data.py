
import os
import zipfile

from pathlib import Path

import requests

def data_request(root_folder: str="data/",
 folder_name: str="pizza_steak_sushi",
 website_link: str="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"):
  root_folder = root_folder
  folder_name = folder_name
  website_link = website_link
  # Setup path to data folder
  data_path = Path(f'{root_folder}') #"data/"
  # data_path = Path("data/")
  image_path = data_path / folder_name #"pizza_steak_sushi"

  # If the image folder doesn't exist, download it and prepare it...
  if image_path.is_dir():
      print(f"{image_path} directory exists.")
  else:
      print(f"Did not find {image_path} directory, creating one...")
      image_path.mkdir(parents=True, exist_ok=True)

  # Download pizza, steak, sushi data
  with open(data_path / f"{folder_name}.zip", "wb") as f:
      request = requests.get(f'{website_link}')
      print(f"Downloading {folder_name} data...")
      f.write(request.content)

  # Unzip pizza, steak, sushi data
  with zipfile.ZipFile(data_path / f"{folder_name}.zip", "r") as zip_ref:
      print(f"Unzipping {folder_name} data...")
      zip_ref.extractall(image_path)

  # Remove zip file
  os.remove(data_path / f"{folder_name}.zip")
  print('All done!')

  train_dir = image_path / 'train'
  test_dir = image_path / 'test'
  print(train_dir, test_dir)
  return train_dir, test_dir
