# coding: utf-8
from torchvision import transforms as T
import torch
import os
import numpy as np
from model import Generator
import cv2
from time import sleep
from downloader import *

# ## Config
image_size =256
selected_attrs =['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
# Dimension of domain labels (1st dataset, i.e. CelebA's selected attrs).
c_dim =5
model_save_dir ='stargan_celeba_256/models'
# The path to download the G weights file.
G_url = 'https://github.com/Tony607/DeepMagicMirror/releases/download/V1.0/200000-G.ckpt'
# Number of conv filters in the first layer of G.
g_conv_dim = 64
# number of residual blocks in G
g_repeat_num = 6
# test model from this step
resume_iters = 200000
# The cv2 CascadeClassifier model path.
CASE_PATH = 'data/custom/haarcascade_frontalface_default.xml'

# ## Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator(g_conv_dim, c_dim, g_repeat_num)
# def print_network(model, name):
#     """Print out the network information."""
#     num_params = 0
#     for p in model.parameters():
#         num_params += p.numel()
#     print(model)
#     print(name)
#     print("The number of parameters: {}".format(num_params))

# print_network(G, 'G')

G.to(device)
print('--------Magic Mirror--------')
print('Project Tutorial link:')
print('https://www.dlology.com/blog/if-i-were-a-girl-magic-mirror-by-stargan/')
print('----------------------------')
print('Model moved to', device)
if device is 'cpu':
    print('----Running on CPU instead of GPU may experience considerable lags.---')
    print('Install cuda support if you have a Nvidia Graphic card.')
print('Loading the trained models from step {}...'.format(resume_iters))


G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iters))
download_if_not_exists(G_path, G_url)
G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

face_cascade = cv2.CascadeClassifier(CASE_PATH)

def crop_face(imgarray, section, margin=20, size=256):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


def MagicMirror(videoFile = 0, setHairColor = 'blond', setMale = False, setYoung = True, sideBySide = True, showZoom = 1):# 0 means the default video capture device in OS
    """
    Args:
    videoFile: leave the default value 0 to use the first web camera, or pass in a video file path.
    setHairColor: one of the three, "black", "blond", "brown".
    setMale: transform into a male? Set to True or False.
    setYoung: transform into a young person? Set to True or False.
    showZoom: default to 4, this factor by which to resize the generated image up before showing on the screen.
    """
    print("Press ESC to exit.")
    # Three hair color to choose from.
    hairColors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair']
    labels = torch.zeros(c_dim)
    # Match the user set hair color and set the label value.
    for i, color in enumerate(hairColors):
        if setHairColor.lower() in color.lower():
            labels[i] = 1
            break
    # Set the 'Male' label.
    if setMale is True:
        labels[3] = 1
    # Set the 'Young' label.
    if setYoung is True:
        labels[4] = 1
    
    video_capture = cv2.VideoCapture(videoFile)
    if ((video_capture == None) or (not video_capture.isOpened())):
        print('No video capture device.')
        return
    
    transform = []
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    winname = "              Deep Magic Mirror | DLology.com"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    main_face = None
    # infinite loop, break by key ESC
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = frame[:, :, ::-1] # CV2'BGR -> RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10
        )
        # generated_frame = None
        if len(faces) > 0:
            main_face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
        if main_face is None:
            x = 0
            y = 0
            w, h = gray.shape
            main_face = (x, y, w, h)
        face_img = crop_face(frame, main_face, margin = 40, size=image_size)
        
        # Pre-process the image
        preprocessed_image = transform(face_img)
        
        # Run the generator to generate the desired image with labels.
        generated = G(preprocessed_image.unsqueeze(0).to(device), labels.unsqueeze(0).to(device))
        
        # Show the generated image.
        generated_frame = ((np.moveaxis(generated.cpu().detach().numpy()[0],[0], [2])+1)/2)[:, ::-1, ::-1]
        if sideBySide:
            real_img = ((np.moveaxis(preprocessed_image.numpy(),[0], [2])+1)/2)[:, :, ::-1]
            concat_frame = np.concatenate((real_img, generated_frame), axis=1)
            if showZoom > 1:
                h, w, _ = concat_frame.shape # w * showZoom, h * showZoom
                concat_frame = cv2.resize(concat_frame, (w * showZoom, h * showZoom), interpolation=cv2.INTER_AREA)
            cv2.imshow(winname, concat_frame)
        else:
            cv2.imshow(winname, generated_frame)

        if cv2.waitKey(5) == 27:  # ESC key press
            print("ESC pressed, existing.")
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # MagicMirror(videoFile = 'data/custom/videos/barack.avi',setHairColor = 'blond', setMale = False, setYoung = True, showZoom = 4)
    MagicMirror(setHairColor = 'blond', setMale = False, setYoung = True, showZoom = 4)
