import os
import sys
from shutil import copyfile
import pandas as pd

def extract_files(folder, c, sample, output):
    data = list()
    for root, dirs, files in os.walk(folder, topdown=True):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith('.jpg'):
                data.append([c, path])
        for name in dirs:
            path = os.path.join(root, name)
            if path.endswith('.jpg'):
                data.append([c, path])
    dataframe = pd.DataFrame(data, columns=['class', 'path'])
    dataframe = dataframe.sample(n=sample)

    os.makedirs(output, exist_ok=True)
    output_image = os.path.join(output, 'images')
    os.makedirs(output_image, exist_ok=True)
    new_path = list()
    for idx in range(len(dataframe)):
        img = dataframe.iloc[idx]['path']
        new_img = os.path.join(output_image, img.split(os.path.sep)[-1])
        copyfile(img, new_img)
        new_path.append(img.split(os.path.sep)[-1])
    dataframe['image'] = new_path
    return dataframe[['class', 'image']].copy()

if __name__ == "__main__":
    cmfd = extract_files("./data/CMFD", 1, 2000, './data/dataset/')
    imfd = extract_files("./data/IMFD", 0, 2000, './data/dataset/')
    dataframe = cmfd.append(imfd)
    dataframe.to_csv('./data/dataset/annotations.csv', index=False)