import cv2
import numpy as np


def showData(data, class_mapping):
    '''
        data : tensorflow.python.data.ops.map_op._ParallelMapDataset
        class_mapping: { 0: "without_mask", 1: "with_mask", 2: "mask_weared_incorrect"}

        show image and label in a batch
    '''
    #get a batch by take(1) after using next(iter())
    # conver to iter and get a batch.
    inputs = next(iter(data.take(1)))

    # inputs = { "image": ..., "bounding_boxes": {"boxes": ..., "classes": ....}}
    images, bounding_boxes = inputs["images"].numpy(), inputs["bounding_boxes"]
    bbox = bounding_boxes["boxes"].numpy()
    classes = bounding_boxes["classes"]
    for i in range(images.shape[0]):

        # get image in images
        img_ = images[i].astype(np.uint8)
        label = classes[i]
        counter = 0
        for box in bbox[i]:
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            
            stringlabel = int(label[counter])
            counter += 1
            labelFinal = class_mapping[stringlabel]

            # point in upper-left
            start_point = (xmin, ymin)
            # point in lower-right
            end_point = (xmax, ymax)
            
            # Draw rectange into img_ width start_point 
            img_ = cv2.rectangle(img_, start_point , end_point, color=(250,0,0), thickness=1)

            # put text to image width font text: 0.5 
            # refer: https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
            img_ = cv2.putText(img_, labelFinal, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,0), thickness=1)

        # show image after draw box and text    
        cv2.imshow("image", img_)

        # enter any button in the keyboard to exit
        cv2.waitKey(0)

    # delete all memory
    cv2.destroyAllWindows()