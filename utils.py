from deepface import DeepFace
import cv2
import numpy as np
import os 
from voyager import Index, Space
from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd 
import yaml
import hashlib
from dataclasses import dataclass, field


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

######################################################################################
    
@dataclass 
class Node():
    nodeType: str # parent/child
    space_id: int # id in Space scale 
    embedding: List[float] # vector embedding 
    closeNodes: List['Node'] = None # for parent node as centroid having its closes child nodes used during cluster expansion 
    parentNode = None      # if child node then its parent's id                      
    
    def isClose(self, otherNode: Union['Node', List[float]], embeddedSpace: voyager.Index)->bool:
        
        if isinstance(otherNode, Node):
            distance = embeddedSpace.get_distance(self.embedding, otherNode.embedding)
        else:
            distance = embeddedSpace.get_distance(self.embedding, otherNode)
        
        status = True if distance < CLOSE_DISTANCE else False
        
        return status, distance 
    
    
    def __eq__(self, otherNode: 'Node') -> bool:
        return self.space_id == otherNode.space_id
    
        
    def getDistantNode(self, embeddedSpace: voyager.Index):
        
        assert self.nodeType == "parent", "Child node can't have list of close nodes"
        sorted_nodes = self.closeNodes.sort(key=lambda node: embeddedSpace.get_distance(self.embedding,
                                                                                        node.embedding))
        distant_node_index = self.closeNodes.index(sorted_nodes[-1])
        distance_to_distant = embeddedSpace.get_distance(self.embedding, self.closeNodes[distant_node_index].embedding)
        
        return distant_node_index, distance_to_distant
    
        

    

class embeddingSearcher():
    
    def __init__(self, data_fldr: str, model_name: str = MODELS[1], 
                 neighbors_num: int = 3, backend_num: int = 2, closenodes_num: int = CLOSENODE_NUM) -> None:
        
        self.data_fldr = data_fldr
        self.neighbors_num = neighbors_num
        self.backend_num = backend_num
        self.model_name = model_name
        self.closenodes_num = closenodes_num
        self.lookup_table = None
        self.embeddedSpace = None
        self.node_collection = {}
        
        try:
            self._loadFiles()
        except Exception as e:
            print(f"Error: {e}") 
            
        
        
    def _loadFiles(self)->None:
        
        """
        Creates Look-Up table for fast searching from the current database
        Unique_ids are created by generating unique id and adding to its end 1, then converted to integer.
        Note, those are only base face images from the main database and groupmates are added iff distance
        condition met. Main image have id pattern (###...###1) and groupmates ids's (###...###2, ###...###3, ....)

        Returns:
            None: 
        """
        
        print("Creating Look-Up Table from the database...")
        
        raw_embeddings, imgs_path = cvtImgs2Embeddings(self.data_fldr , 1,  self.backend_num)
        
        index = Index(Space.Cosine, num_dimensions=DIM_SIZES[self.model_name]) 
        
        num_entities = len(raw_embeddings)
        unique_ids_parent = [generate_unique_id(img_path) for img_path in imgs_path]
        
        self.lookup_table = pd.DataFrame(data=imgs_path, columns=['Image path'], index=unique_ids_parent) 
        
        for ind in range(num_entities):
            index.add_item(vector=raw_embeddings[ind], id=unique_ids_parent[ind])
            
        self.embeddedSpace = index 
        self._initNodeCollection(raw_embeddings, unique_ids_parent)
        self.lookup_table.to_csv("./Lookup_Table.csv")

        
    def _initNodeCollection(self, raw_embeddings: List[List[float]], unique_ids_parent: List[int]):
        
        """
            Initializes parent nodes and append to node collection to keep track 
            nodes in embedded Space

        """
        
        for embedding, id in zip(raw_embeddings, unique_ids_parent):
            parent_node = Node(nodeType="parent", space_id=id, embedding=embedding)
            self.node_collection[id] = parent_node
        
        print(f"Initilized {self.embeddedSpace.num_elements} parent nodes")
            

    def findNearestNeighbors(self, frame: np.array, draw_bbox = False, face_distance: float=0.4, extendSpace: bool=False)->List[Dict[str, Any]]:
        
        """
            Class method used to find faces in the frame along with its matched faces from the 
            database
            If extendSpace is True, the face embedding will be attached to the closes parent Node
            and the distance should be small. 
            
        Args:
            frame (np.array): Frame obtianed from the stream or camera 

        Returns:
            List[Dict[str, Any]]: Each element in the list represents  
        """
    
        detected_faces = DeepFace.represent(frame,  model_name=MODELS[1], enforce_detection = False , 
                                            detector_backend = BACKENDS[self.backend_num])
        
        faces_list = list()
        
        for face in detected_faces: # for each detected face in a frame find matches in database
            face_embedding, face_coords = face['embedding'], face['facial_area'] 
            
                    
            if face['face_confidence'] > FACE_CONFIDENCE:
                
                neighbors, distances = self.embeddedSpace.query(face_embedding, k = self.neighbors_num)
                
                found_matches_path = []
                
                for neighbor_id in neighbors:          
                    matched_face_path = self.lookup_table.loc[neighbor_id].values
                    found_matches_path.append(matched_face_path)
                    
                face_dict = {"detected_face_coord": face_coords, "path_to_matches": found_matches_path,
                             "distances": distances}
                
                faces_list.append(face_dict)
                
                if extendSpace:
                    closestParentID = self.embeddedSpace.get_vector(neighbors[0])
                    parentNode = self.node_collection[closestParentID]
                    self._addNewEmbeedingOnDistance(parentNode, face_embedding) # Add as a new embedding if it gives small distance
                    
                print(f"Face detected with ids: {neighbors}")
        
        if draw_bbox:  # Draw bboxes around faces and matched faces from the database
            self._drawBboxes(frame, faces_list, face_distance)
        
        return faces_list  
             
            
            

    def _drawBboxes(self, frame: np.array, faces_list: List, face_distance: float):
        
        """
            Draw bboxes on detected faces with its top 1 mathced face in the Database 
            and his name respectively

        Args:
            frame (np.array): current frame
            faces_list (List): List of detected faces with information like coordinates and mathced 
                                face image path 
        """
        
        for face in faces_list:
            
            face_coords, found_matches_path = face["detected_face_coord"], face["path_to_matches"]
            
            distance= face["distances"][0]
            best_match_path = os.path.join(self.data_fldr, found_matches_path[0][0])
        
            label = os.path.basename(best_match_path).split('.')[0]
            target_img = cv2.imread(best_match_path)
            
            
            try:
                assert face_coords['w'] > 0 and face_coords['h'] > 0, "Positive coordinates are expected"
                
                draw_coord = face_coords['x']+face_coords['w'], face_coords['y'], \
                                face_coords['x']+2*face_coords['w'], face_coords['y']+face_coords['h']
                                
                if distance < face_distance: # Assign known face
                    
                    target_img = cv2.imread(best_match_path)
                    draw_text(frame, label, pos=(face_coords['x'], face_coords['y']-45))
                    
                else:  # Assign unknown face
                    
                    target_img = cv2.imread("./unknown.jpeg")
                    draw_text(frame, "Unknown", pos=(face_coords['x'], face_coords['y']-45))
                    
                resized_target = cv2.resize(target_img, (face_coords['w'], face_coords['h']))
                
                if draw_coord[0]>0 and (draw_coord[2]<frame.shape[1]) and draw_coord[1]>0 and draw_coord[3]<frame.shape[0]:

                    frame[draw_coord[1]:draw_coord[3], draw_coord[0]:draw_coord[2]] = resized_target
                    cv2.rectangle(frame, (face_coords['x'], face_coords['y']), (face_coords['x']+face_coords['w'], face_coords['y']+face_coords['h']), (255, 255, 0), 2) 
                else:
                    print("Resized image dimensions exceed bbox dimensions. Skipping drawing.")
                    
            except Exception as e:
                print(e)     

    
    
    
    def _addNewEmbeedingOnDistance(self, parentNode: Node, child_embedding: List[float]):
    
        isChild, distance_to_new = parentNode.isClose(child_embedding, self.embeddedSpace) # is Close enough ?
        child_num = len(parentNode.closeNodes)
        
        # Creation of new child node 
        newChildID = int(str(parentNode.space_id) + str(len(parentNode.closeNodes)))
        newChild = Node(nodeType="child", space_id=newChildID, 
                            embedding=child_embedding, parentNode=parentNode.space_id)
        
        if isChild and child_num < self.closenodes_num:
            # Adding to the embedded Space and node collection 
            parentNode.closeNodes.append(newChild)
            self.embeddedSpace.add_item(vector=child_embedding, id=newChildID)
            
        elif isChild and child_num > self.closenodes_num:
            distant_node_index, distance_to_distant = parentNode.getDistantNode(self.embeddedSpace)

            if distance_to_distant > distance_to_new: # replace distant node to the new 
                
                # Update old node with new in embedded space and node collection
                
                # TODO: Consider how to remove the old one !!!
                
                del parentNode.closeNodes[distant_node_index] 
                parentNode.closeNodes.insert(distant_node_index, newChild)
        
        else:
            del newChild
        



