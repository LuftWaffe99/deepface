import cv2
from utils import embeddingSearcher
                  

face_finder = embeddingSearcher("./Faces")

if __name__ == "__main__":
    
    stream = cv2.VideoCapture(0)

    while True:
        ret, frame = stream.read()
        if not ret:
            break 
        
        image = cv2.imread("./Abzal3.jpg")
        # face_finder.checkFace(image)
        result = face_finder.findNearestNeighbors(frame, draw_bbox=True)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    stream.release()
    cv2.destroyAllWindows()
