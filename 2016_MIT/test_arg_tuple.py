import argparse 

def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' Script read both depth and rgb frames from kinetic 
        and save them in Assigned Folder''')
    # Add arguments

    parser.add_argument('-d', "--descriptor", help="Use - HOG",
            default="HOG")
    parser.add_argument('-HOGw',"--width", help="width of image to HOG")
    parser.add_argument('-HOGh',"--height", help="height of image to HOG")
    args = parser.parse_args()

    # Assign args to variables
    width, height = args['width'] , args['height']
    des_type = args['descriptor']
    

    ###port = args.port[0].split(",")
    ###keyword = args.keyword
    # Return all variable values
    return des_type, width, height

def main():
    #des_type, width, height = get_args()
    #print 'width is %s and height is %s'%(width, height)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-n', "--negpath", help="Path to negative images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]
    des_type = args["descriptor"]
    print pos_im_path
    print neg_im_path
    print des_type

if __name__=="__main__":
	main()