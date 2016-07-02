########################################################################

import os
import argparse

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic
        and save them in Assigned Folder''')
    # Add arguments
    parser.add_argument(
        '-fn', '--folder_name', type=str, \
        help='Object or Event as Folder name', required=True)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    folder_name = args.folder_name
    # Return all variable values
    return folder_name


def main():
    origin_file_name = os.listdir((os.getcwd()))
    
    # disgard .DSTORE or else ...
    origin_file_name = [s for s in origin_file_name if '.jpg' in s or '.png' in s]

    # Get Customized Filename
    folder_name=get_args()
    
    # New_File_Name for list
    new_file_name = \
    [folder_name+ str(i)+'.png' \
    for i in range(len(origin_file_name))]

    # Substitude the file name
    for i in range(len(origin_file_name)):
        os.rename(origin_file_name[i], new_file_name[i])

    file = open('background.txt', 'w')
    for i in new_file_name:
        file.write(i+'\n')
    file.close()

if __name__=='__main__':
	main()