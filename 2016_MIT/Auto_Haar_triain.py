


#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
#==============================================================================
#
# Notice
# - This recipe would automate the training process powered by OpenCV
# - This recipe works only in Mac OS or Linux
# - Before fire this recipe, check your folder strcuture as next section
# - If you dont have BG.file, use another recipe rename_img_file_for_training
# - After run this recipe, you will get a cascade.xml in the result folder
# - It will also generate the command histroy file for record
# - This cascade.xml file could be used by the opencv CascadeClassifier
#
#==============================================================================
# Folder Struture
#==============================================================================
#
# Main Folder
#  |
#  -- positive_name Folders(i)
#        |
#        -- positive_name1.png
#        -- positive_nameN.png .... where N belongs int.
#        -- BG.txt ..... which contains list of p_img names
#
#  -- negative Folders(j)
#        |
#        -- bg1.png
#        -- bgN.png ..where N belongs int.                                
#        -- BG.txt ..... which contains list of n_img names
#  -- traing_fold
#
#==============================================================================

import os
import argparse

def get_args():
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Automate training process''')

    # Add arguments of which positive_fold 
    parser.add_argument(
        '-pn', '--pf_name', type=str, 
        help='Object or Event as Folder name', required=True)

    # Add arguments of which positive_fold 
    parser.add_argument(
        '-nn', '--nf_name', type=str, 
        help='Object or Event as Folder name', required=True)

    parser.add_argument(
        '-tn', '--tf_name', type=str, 
        help='The folder name you want to create', required=True)  

    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    pf_name , nf_name, tf_name = args.pf_name , args.nf_name, args.tf_name
    # Return all variable values
    return pf_name , nf_name, tf_name

def main():
    # get to the main folder
    ##os.path.dirname(os.getcwd())
    pf_name , nf_name, tf_name = get_args()

    # return list object 
    with open(pf_name+'/BG.txt') as f:
        pf_content = f.readlines()

    # return list object 
    with open(nf_name+'/BG.txt') as f:
        nf_content = f.readlines()    

    # create the training folder 
    os.system('mkdir %s'%(tf_name))

    # Start opencv_createsamples for all positive files
    for p_file in pf_content:
    	# Doc Watch out /n
        p_file = p_file.split('\n')[0]

        # Data Augmentation Code Generator
        code ='opencv_createsamples -img %s -bg %s/BG.txt -info %s/des_%s.txt'\
        %(pf_name+'/'+p_file, nf_name, tf_name, p_file.split('.')[0])

        # Excute the code
        os.system(code)

    # Cat all des files in to positive.txt Then export .vec file
    os.system('cat %s/des*.txt -> %s/positive.txt'%(tf_name,tf_name))
    os.system('opencv_createsamples -info %s/positive.txt -vec %s/output_vec.vec'\
    	%(tf_name, tf_name))

    # mkdir of result to record cascade.xml
    os.system('mkdir %s/result'%(tf_name))

    # Start to training but this stage is unstable ... 
    os.system('opencv_traincascade -data %s/result/ '\
        '-vec %s/output_vec.vec -bg %s/bg.txt -numPos %d -numNeg %d')%(tf_name,\
        tf_name, nf_name,len(pf_content),len(nf_content))

    # 
    os.system('history > command_history.txt')

if __name__=='__main__':
    main()