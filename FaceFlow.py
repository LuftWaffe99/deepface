from deepface import DeepFace
import numpy as np
import os 
from voyager import Index, Space
import voyager
from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd 
import yaml
import hashlib
import cv2
from dataclasses import dataclass, field   
from utils import generate_unique_id, draw_text, cvtImgs2Embeddings


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


FACE_CONFIDENCE = config['THRESHOLD']['Face_Confidence']
CLOSE_DISTANCE = config['THRESHOLD']['newNode_Dist']
CLOSENODE_NUM = config['THRESHOLD']['closeNode_Num']
BACKENDS = config['BACKENDS']
MODELS = config['MODELS']
DIM_SIZES = config['DIM_SIZES']



@dataclass 
class Node():
    nodeType: str # parent/child
    space_id: int # id in Space scale 
    embedding: List[float] # vector embedding 
    closeNodes: List['Node'] = field(default_factory=list) # for parent node as centroid having its closes child nodes used during cluster expansion 
    parentNode: int = None      # if child node then its parent's id                      
    
    def isClose(self, otherNode: Union['Node', List[float]], embeddedSpace: Index)->bool:
        
        """
            Based on distance threshold identifies whether the child/parent node is close 
            to another node or embedding
            
        Args:
            otherNode (Union[&#39;Node&#39;, List[float]]): Node type input (can be parent or child type)
            embeddedSpace (voyager.Index): embedding space/index

        Returns:
            bool: returns boolean value 

        """
        
        if isinstance(otherNode, Node):
            distance = embeddedSpace.get_distance(self.embedding, otherNode.embedding)
        else:
            distance = embeddedSpace.get_distance(self.embedding, otherNode)
        
        status = True if distance < CLOSE_DISTANCE else False
        
        return status, distance 
    
    
    def __eq__(self, otherNode: 'Node') -> bool:
        return self.space_id == otherNode.space_id
    
        
    def getDistantNode(self, embeddedSpace: Index)->Tuple[int, float, int]:
        
        """
            For the parent node it finds the most farthest child embedded node (index in parent's child list,
            distance, and its space id) 

        Returns:
            Returns tuple of information regarding the the most farthest point appended in parent's list: 
        """
        
        assert self.nodeType == "parent", "Child node can't have list of close nodes"
        sorted_nodes = self.closeNodes.sort(key=lambda node: embeddedSpace.get_distance(self.embedding,
                                                                                        node.embedding))
        distant_node_index = self.closeNodes.index(sorted_nodes[-1])
        distance_to_distant = embeddedSpace.get_distance(self.embedding, self.closeNodes[distant_node_index].embedding)
        
        return distant_node_index, distance_to_distant, self.closeNodes[distant_node_index].space_id
    
        

    

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
        self._initNodeCollection(raw_embeddings, unique_ids_parent) # initialization of parent node list
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
            

    def findNearestNeighbors(self, frame: np.array, draw_bbox = False, face_distance: float=0.4, extendSpace: bool=True)->List[Dict[str, Any]]:
        
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
                    try:         
                        matched_face_path = self.lookup_table.loc[neighbor_id].values
                    except Exception as e: # handle child embedding case (not specified in Look-Up table)
                        padding_index = -len(str(self.closenodes_num))
                        padding_removed = int(str(neighbor_id)[:padding_index])
                        matched_face_path = self.lookup_table.loc[padding_removed].values
                        
                    found_matches_path.append(matched_face_path)
                    
                face_dict = {"detected_face_coord": face_coords, "path_to_matches": found_matches_path,
                             "distances": distances}
                
                faces_list.append(face_dict)
                
                if extendSpace: # Add as a new embedding if it gives small distance
                    
                    closestParentID = self._extractParentNodeID(neighbors)
                    
                    if closestParentID is not None:
                        parentNode = self.node_collection[closestParentID]
                        self._addNewEmbeedingOnDistance(parentNode, face_embedding) 
                        
                # print(f"Face detected with ids: {neighbors}")
        
        if draw_bbox:  # Draw bboxes around faces and matched faces from the database
            self._drawBboxes(frame, faces_list, face_distance)
        
        return faces_list  
             
    
    
    def _extractParentNodeID(self, neighbors: List[int]) -> int:
        
        parentFound = None
        
        for closeNode in neighbors:
            if closeNode in [item.space_id for item in self.node_collection.values()]:
                parentFound = closeNode
                break
        # print(parentFound)
        return parentFound
            
            

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
        
        """
            Add new embeddings as a child node for the closest parent Node if the number of chilld 
            nodes of that parent node is not oveflowed (closenodes_num param). If the number is more then find the distant 
            child node and replace it with new Node if the distance is smaller 

        Args:
            parentNode (Node): _description_
            child_embedding (List[float]): _description_
        """
        
        isChild, distance_to_new = parentNode.isClose(child_embedding, self.embeddedSpace) # is Close enough ?
        child_num = len(parentNode.closeNodes)
        
        # Creation of new child node 
        padding = len(str(self.closenodes_num))
        padded_id = str(child_num).zfill(padding)
        newChildID = int(str(parentNode.space_id) + padded_id)
        newChild = Node(nodeType="child", space_id=newChildID, 
                        embedding=child_embedding, parentNode=parentNode.space_id)
        
        if isChild and child_num < self.closenodes_num:
            # Adding to the embedded Space and node collection 
            parentNode.closeNodes.append(newChild)
            self.embeddedSpace.add_item(vector=child_embedding, id=newChildID)
            print(f"Added new child for {parentNode.space_id}")
            
        elif isChild and child_num > self.closenodes_num:
            
            distant_node_index, distance_to_distant, child_id = parentNode.getDistantNode(self.embeddedSpace)

            if distance_to_distant > distance_to_new: # replace distant node to the new 
                
                # Update old node with new in embedded space and node collection    
                del parentNode.closeNodes[distant_node_index] 
                parentNode.closeNodes.insert(distant_node_index, newChild)
                
                self.embeddedSpace.mark_deleted(child_id)
                self.embeddedSpace.add_item(vector=child_embedding, id=newChildID)
                print(f"Added new child for {parentNode.space_id} and removed {child_id}")
        else:
            del newChild
        
        
        
        def reloadSpace(self):
            pass
            # TODO: after some time when memory is overloaded, create new Space 
            # copy from the node_collection and delete the old one 