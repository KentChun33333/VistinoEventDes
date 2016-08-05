import argprase


def get_args():
    '''use for single py'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description=''' use -conf to train model-detectors''')
    # Add arguments
    parser.add_argument('-conf', "--conf_path", help="Path to conf_hub",
            required=True)
    parser.add_argument('-img', "--img", help="Path to img",
            required=True)    
    # Parses
    args = parser.parse_args()
    # Assign args to variables
    conf_path = args.conf_path
    img_path = arrgs.img
    # Return all variable values
    return conf_path, img_path


parser = argparse.ArgumentParser(description='Prepare Training Data From Folders')

parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix



def training_data(videol, )









