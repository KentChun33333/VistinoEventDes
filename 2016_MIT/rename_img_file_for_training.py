
#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
#
# Operation Notice
# - This recipe would rename all image_files in Current Working Folder
# - It would also generate a new BG.file in current folder 
# - This recipe is focus on pre-process for following OpenCV Haar training 
# - Be aware of all raw files' name would be changed after this recipe
# 
#==============================================================================

import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--folder_name', type=str,
        help='add the New Name for this set of images', required=True)

    args = parser.parse_args()
    return args.folder_name


def main():
    # cuurent working folder
    rawName = os.listdir((os.getcwd()))
    rawName = [s for s in rawName if '.jpg' in s or '.png' in s]

    # Get Customized Filename
    folder_name=get_args()
    
    # New_File_Name for list
    new_file_name = [folder_name+ str(i)+'.png' \
    for i in range(len(rawName))]

    # Substitude the file name
    for i in range(len(rawName)):
        os.rename(rawName[i], new_file_name[i])

    with open('BG.txt', 'w') as f:
        for i in new_file_name:
            f.write(i+'\n')

if __name__=='__main__':
	main()