import os
import glob
import shutil
from pathlib import Path

def get_list_of_paths(path: str):
    '''
    Returns a sorted list of paths of files inside a folder path
    '''
    return sorted([image for image in Path(path).glob('*') 
                              if image.name != '.DS_Store'])

def get_list_of_file_names(path: str):
    '''
    Returns a sorted list of paths of names of files inside a folder path
    '''
    return sorted([os.path.basename(image) for image in Path(path).glob('*') 
                              if image.name != '.DS_Store'])

def create_empty_folder(path:str):
        '''
        Creates empty fodlers if they don't exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        

def empty_directory(path:str):
    '''
    Deletes everything in a directory
    '''
    shutil.rmtree(path)
    os.makedirs(path)
    file_path = os.path.join(path, '.gitkeep')

    with open(file_path, "w", encoding="utf-8") as f:
        pass
