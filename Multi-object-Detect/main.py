
# Author : Kent Jin-Chun Chiu
# Multi-object detection



import argparse
import cv2


def get_args():
    parser = argparse.ArgumentParser(
        description=''' use -conf to train model detectors''')
    parser.add_argument('-conf', "--conf_path", help="Path to conf_hub",
            required=True)
    args = parser.parse_args()
    conf_path = args.conf_path
    return conf_path

def main():
	pass

if __name__=='__main__':
	main()
