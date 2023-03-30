#Made by Vasilii Pustovoit, CITA, March 2023
import os
import numpy as np

#Relevant filename-related variables
file_extension = 'avi'
input_dir='./shock_velocity/'
output_dir = './'
output_name = 'smoothing_lengths' #Please omit the file extension

#Assigning/changing variables
files = os.listdir(input_dir)
output_name = output_name + '.' + file_extension

#Make a movie {{{
print('Creating ' + output_name + '...')
if file_extension == 'gif': #{{{
    import imageio

    plot_files=[]
    for i in range(0, len(files)-1):
        plot_files=np.append(plot_files, input_dir +files[i])
    
    # Sort the list of files so that the gif is created in order
    plot_files.sort()
    
    # Create an empty list to store the images
    images = []
    
    # Loop over all the plots and add each to the list of images
    i=0
    for plot in plot_files:
        images.append(imageio.imread(plot))
        #print(i)
        i=i+1
    # Save the list of images as a gif
    imageio.mimsave(output_dir+output_name, images)
#}}}

elif file_extension == 'avi': #{{{
    import cv2
    # Get the dimensions of the first image to use as the video frame size
    frame = cv2.imread(os.path.join(input_dir, os.listdir(input_dir)[0]))
    height, width, channels = frame.shape
    
    # Create a VideoWriter object to write the frames into a video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for mp4 format
    video = cv2.VideoWriter(os.path.join(output_dir, output_name), fourcc, 30, (width, height))
    
    # Loop over all the plots and add each to the video
    i = 0.0
    for file in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file)
        i = i + 1/len(os.listdir(input_dir))
        percent = int(i * 100)
        print(f"\rCreating movie... {percent}%", end="")
        frame = cv2.imread(filepath)
        video.write(frame)
    
    # Release the VideoWriter object and close the video file
    video.release()
    cv2.destroyAllWindows()
    print("\rCreating movie... Done!")
#}}}

else:
    print("Error: this file extension is not supported!")
#}}}
