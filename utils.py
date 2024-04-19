from deepface import DeepFace
import cv2
import numpy as np
import os 
from voyager import Index, Space
from typing import List
import yaml
import hashlib
import voyager


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


FACE_CONFIDENCE = config['THRESHOLD']['Face_Confidence']
CLOSE_DISTANCE = config['THRESHOLD']['newNode_Dist']
CLOSENODE_NUM = config['THRESHOLD']['closeNode_Num']
BACKENDS = config['BACKENDS']
MODELS = config['MODELS']
DIM_SIZES = config['DIM_SIZES']




def generate_unique_id(img_path: str) -> int:
    """
    Generate a 16-digit unique ID for each image based on its filename.

    Args:
        img_path (str): Filename of the image.

    Returns:
        int: 16-digit integer unique number based on the image filename.
    """
    encoded_string = img_path.encode('utf-8')

    
    hash_object = hashlib.sha256(encoded_string)
    hex_digest = hash_object.hexdigest()

    truncated_hex_digest = hex_digest[:16]  # Get the first 16 characters
    unique_id = int(truncated_hex_digest, 16)  # Convert hex to integer
    
    unique_id = unique_id % (10**16)  # Keep only the last 16 digits

    return unique_id



def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=3, 
              font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0))->None:
    """
    Print text with background color specified in arguments 

    Args:
        img (np.array): frame or image
        text (str): text to be printed
        font (_type_, optional): font size. Defaults to cv2.FONT_HERSHEY_PLAIN.
        pos (tuple, optional): position of text to be printed. Defaults to (0, 0).
        font_scale (int, optional): font scale. Defaults to 2.
        font_thickness (int, optional): text font thickness. Defaults to 2.
        text_color (tuple, optional): color of the text. Defaults to (0, 255, 0).
        text_color_bg (tuple, optional): background color of the text. Defaults to (0, 0, 0).

    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (pos[0], pos[1]-5), (x + text_w, y + text_h+5), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)




def detect_recognize(frame: np.array, db_path: str = "Faces", model_num: int = 0)->None:
    
    """
        Detects and Recognize faces on frame
    Args:
        frame (np.array): numpy frame
        db_path (str, optional): path to dataset of face images. Defaults to "Faces".
        model_num (int, optional): select from the MODELS list. Defaults to 0.
    """
    
    result_df = DeepFace.find(img_path=np.array(frame), db_path=db_path, model_name=MODELS[model_num],
                                    enforce_detection=False, threshold=0.4, silent=True)
    
    isDetected = True if (len(result_df[0].index)) > 0 else False
    
    if isDetected:

        result_df = result_df[0].head(3) # take at most top 3 matches
        source_bboxs = result_df[['source_x', 'source_y', 'source_w', 'source_h']].values
        target_bboxs = result_df[['target_x', 'target_y', 'target_w', 'target_h']].values
        identities = result_df[['identity']].values
        
        for ind, bbox in enumerate(source_bboxs): # iterate through 
            
            target_bbox = target_bboxs[ind, :]
            target_img_path = identities[ind]
            label = os.path.basename(target_img_path[0]).split('.')[0]

            if (target_bbox[2]-target_bbox[0]) > 0 and (target_bbox[3]-target_bbox[1]) > 0:
                
                target_img = cv2.imread(target_img_path[0])
                resized_target = cv2.resize(target_img, (target_img.shape[0] // 3, target_img.shape[0] // 3))
                resizedH, resizedW = resized_target.shape[:2]
                # Draw resized_target on the right corner of the detected boun  ding box rectangle
                frame[bbox[1]:bbox[1]+resizedH, bbox[0]+bbox[2]:bbox[0]+bbox[2]+resizedW] = resized_target
                
                draw_text(frame, label, pos=(bbox[0], bbox[1]-30))
                
            # Draw the original bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 0), 2) 
            
            
    
        
            

def cvtImgs2Embeddings(db_path: str, model_name: int = 1, backend_num: int = 2)->List[List[float]]:
    
    """
    Get images from the databases and encode them

    Returns:
        List[List[float]]: Returns list of embedded vectors represented as a list of components
    """
    
    embedded_imgs = list()
    imgs_name = list()
    
    for filename in os.listdir(db_path):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(db_path, filename)
            face_name = os.path.basename(img_path).split('.')[0] # extract name 
            
            embedded_img = DeepFace.represent(img_path, model_name=MODELS[model_name], detector_backend=BACKENDS[backend_num])
            
            imgs_name.append(filename)
            embedded_imgs.append(embedded_img[0]['embedding'])
    
    
    return embedded_imgs, imgs_name


 



