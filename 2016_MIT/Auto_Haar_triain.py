


###########################################################################
# @Author : Kent Chiu (kentchun33333@gmail.com)                           #
###########################################################################
# Operation Notice :                                                      #
# Work everything at the Main Folder layer                               #
###########################################################################
# Folder Struture :                                                       #
# @@@@@@@@@@@@                                                            #
# Main Folder                                                             #
#  |                                                                      #
#  -- positive_name Folders(i)                                            #
#        |                                                                #
#        -- positive_name1.png                                            #
#        -- positive_nameN.png .... where N belongs int.                  #
#        -- background.txt ..... which contains list of p_img names       #
#  -- negative Folders(j)                                                 #
#        |                                                                #
#        -- bg1.png                                                       #
#        -- bgN.png ..where N belongs int.                                #
#        -- background.txt ..... which contains list of n_img names       #
#  -- traing_fold                                                         #
#                                                                         #
###########################################################################

import os
import argparse

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Automate training process''')
    # Add arguments of which positive_fold 
    parser.add_argument(
        '-pn', '--pf_name', type=str, \
        help='Object or Event as Folder name', required=True)
    # Add arguments of which positive_fold 
    parser.add_argument(
        '-nn', '--nf_name', type=str, \
        help='Object or Event as Folder name', required=True)
    parser.add_argument(
        '-tn', '--tf_name', type=str, \
        help='Object or Event as Folder name', required=True)            
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
    # 
    with open(pf_name+'/background.txt') as f:
        pf_content = f.readlines()
    # 
    with open(nf_name+'/background.txt') as f:
        nf_content = f.readlines()    
    #
    os.system('mkdir %s'%(tf_name))
    # Start opencv_createsamples for all positive files
    for p_file in pf_content:
    	# Doc Watch out /n
        p_file = p_file.split('\n')[0]
        code ='''opencv_createsamples -img %s -bg %s/background.txt -info %s/des_%s.txt'''\
        %(pf_name+'/'+p_file, nf_name, tf_name, p_file.split('.')[0])
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

if __name__=='__main__':
    main()



## Step 1
#for i in range(143):
#	os.system('opencv_createsamples -img RHand/Rhand%s.png -bg background.txt -info output_des%s.txt'%(i,i))#
## Step 2
#os.system('cat output_des*.txt -> positive.txt')#

#os.system('opencv_createsamples -info positive.txt -vec output_vec.vec')
#os.system('mkdir result')
#os.system('opencv_traincascade -data result/ -vec output_vec.vec -bg bg.txt -numPos 715 -numNeg 5')






