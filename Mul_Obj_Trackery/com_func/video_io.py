
import cv2

def video_saving(fileName, fps, imgSequence):
    height, width, channels = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        out.write(image) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

def dep_video_saving(fileName, fps, imgSequence):
    height, width = imgSequence[0].shape    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(fileName, fourcc, fps, (width, height))    
    for image in imgSequence:
        img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) # avoid 3|4 ERROR
        out.write(img) # Write out frame to video    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()