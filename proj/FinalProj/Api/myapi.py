from fastapi import FastAPI, File, UploadFile
import os
from pathlib import Path
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import random
import cv2
import time
import requests
from collections import OrderedDict,namedtuple

app = FastAPI()

# Get the desktop path
desktop_path = str(Path.home() / "Desktop")

# Create a new folder on the desktop
new_folder_path = os.path.join(desktop_path, "ModelsTEST")
os.makedirs(new_folder_path, exist_ok=True)

# Load the ONNX model
onnx_model_path = "deeplab_model.onnx"
session1 = ort.InferenceSession(onnx_model_path)

color_to_class = {
    0: (255, 197, 25),  # Forklift
    1: (140, 255, 25),  # Rack
    2: (140, 25, 255),  # Crate
    3: (226, 255, 25),  # Floor
    4: (255, 111, 25),  # Railing
    5: (255, 25, 197),  # Pallet
    6: (54, 255, 25),   # Stillage
    7: (25, 255, 82),   # iwhub
    8: (25, 82, 255),   # Dolly
    9: (0, 0, 0)        # Background
}

def test_onnx_model_on_single_image(session, image, color_map):
    # Get the input and output names for the ONNX model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Preprocess the image
    image = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW format
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform inference
    outputs = session.run([output_name], {input_name: image})
    
    # Get the predicted class labels
    predicted = np.argmax(outputs[0], axis=1).squeeze(0)

    # Initialize an empty RGB image
    height, width = predicted.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each class label to the corresponding RGB value
    for class_id, rgb_value in color_map.items():
        rgb_image[predicted == class_id] = rgb_value

    # Convert the numpy array to a PIL Image and return
    return Image.fromarray(rgb_image)

available_models = [
    {
        "model_id": "1",
        "name": "ResNet50 + DeepLabV3Plus",
        "description": "Industry specific image semantic segmentation. returns image segmented",
        "input_type": "image",
        "return_type":"image",
        "path":"/Segment"
        
    },
    {
        "model_id": "2",
        "name": "YOLOv5m-bbxs",
        "description": "Industry specific object detection. returns bounding boxes with respective class and accuracy ",
        "input_type": "image",
        "return_type":"JSON file"
    },
    {
        "model_id": "3",
        "name": "YOLOv5m-image",
        "description": "Industry specific object detection.returns the same image with bounding boxes overlayed on it",
        "input_type": "image",
        "return_type":"image"
    }
]

@app.get("/Models")
async def model_listing():
    return {"Available models" :available_models}


@app.post("/Segment")
async def Image_Segmentation(file: UploadFile = File(...)):
    # Open the uploaded image
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')

    # Perform segmentation using the ONNX model
    output_image = test_onnx_model_on_single_image(session1, image, color_to_class)

    # Define the fixed filename and save path
    fixed_filename = "segmentation_test.png"
    save_path = os.path.join(new_folder_path, fixed_filename)

    # Save the segmented image
    output_image.save(save_path)

    return {"Status": "Success", "filename": fixed_filename, "location": save_path}

w="yoloHigher.onnx"

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def array_to_dict(array_of_arrays):
    if isinstance(array_of_arrays, np.ndarray):
        array_of_arrays = array_of_arrays.tolist()  # Convert NumPy array to list of lists
    result_dict = {str(index): sub_array for index, sub_array in enumerate(array_of_arrays)}
    return result_dict

session = ort.InferenceSession(w)

names = [ 'Forklift', 'Rack' , 'Crate' , 'Floor' , 'Railing' , 'Pallet' , 'Stillage' , 'iwhub' , 'Dolly'  ]
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

@app.post("/Boundingboxes")
async def return_bbxs(file: UploadFile = File(...)):
    pil_image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Convert the PIL image to a NumPy array
    img = np.array(pil_image)
    
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    

    inp = {inname[0]:im}
    
    outputs = session.run(outname, inp)[0]
    
    return {"Bounding Boxes" :array_to_dict(outputs)}

@app.post("/Boundingboxes_onimg")
async def return_bbxs_onimg(file: UploadFile = File(...)):
    
    pil_image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Convert the PIL image to a NumPy array
    img = np.array(pil_image)

    # Convert RGB to BGR (since OpenCV uses BGR by default)
    #img = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    

    inp = {inname[0]:im}
    
    outputs = session.run(outname, inp)[0]
    
    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,1)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

    output_image=Image.fromarray(ori_images[0])
    
    filename = "object_detection_test.png"
    save_path = os.path.join(new_folder_path,filename)
    
    
    output_image.save(save_path)

    return {"Status": "Success", "filename": filename, "location": save_path}

