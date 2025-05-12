import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from model import GrayscaleNet

classes = ['b_adag', 'ba', 'bi', 'bo', 'd', 'd_dund', 'ge', 'gedes', 'gi', 'gu', 'h', 'l', 'l_adag', 'm', 'm_adag', 'n', 'n_adag', 'num', 'orhits', 'ou_adag', 'r', 'r_adag', 's', 'sh', 'shilbe', 'shud', 'suul', 'ts', 'y', 'z']


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def crop_image(binary):
    coords = np.column_stack(np.where(binary < 255))
    if coords.size > 0:
        x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
        y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
        binary = binary[y_min:y_max,  x_min:x_max]
    return binary 

def segment_cols(binarized_img):
    vertical_projection = np.sum(binarized_img, axis=0)
    threshold = np.max(vertical_projection) * 0.001
    column_indices = np.where(vertical_projection > threshold)[0]
    columns = []
    current_column = [column_indices[0]]

    for i in range(1, len(column_indices)):
        if column_indices[i] - column_indices[i - 1] > 5:
            columns.append((current_column[0], current_column[-1]))
            current_column = [column_indices[i]]
        else:
            current_column.append(column_indices[i])

    columns.append((current_column[0], current_column[-1]))

    column_images = []
    for i, (start, end) in enumerate(columns):
        column_img = binarized_img[:, start:end] 
        column_images.append(column_img)

    return column_images

def segment_words(binarized_img):
    words = []
    _, binary = cv2.threshold(binarized_img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1]) 
        
    for x, y, w, h in bounding_boxes:
        segment = binarized_img[y:y+h, x:x+w]
        cropped_word = crop_image(segment)
        words.append(cropped_word) 
        
    return words

def segment_characters(image):
    chars = []
    vertical_projection = np.sum(image, axis=0)
    threshold = np.max(vertical_projection) * 0.5
    backbone_indices = np.where(vertical_projection > threshold)[0]
    all_indices = np.arange(image.shape[1])
    non_backbone_indices = np.setdiff1d(all_indices, backbone_indices)

    backbone_pixels = [(y, x) for x in non_backbone_indices for y in range(image.shape[0]) if image[y, x] == 0]

    for (y, x) in backbone_pixels:
        image[y, x] = 255  


    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV) 
    horizontal_projection = np.sum(binary, axis=1)
    threshold = np.max(horizontal_projection) * 0.1
    split_positions = np.where(horizontal_projection < threshold)[0]

    splits = []
    prev = 0
    for pos in split_positions:
        if pos - prev > 2:
            splits.append(pos)
        prev = pos

    if len(splits) == 0 or splits[-1] != binary.shape[0]:
        splits.append(binary.shape[0])

    split_images = []
    start = 0
    for split in splits:
        char_img = binary[start:split, :]

        if char_img.shape[0] > 1:
            for (y, x) in backbone_pixels:
                if start <= y < split and 0 <= x < binary.shape[1]:
                    char_img[y - start, x] = 255  # Adjust y position

            split_images.append(char_img)

        start = split
    min_height = 3
    for char_img in split_images:
        inverted_img = cv2.bitwise_not(char_img)
        cropped_char = crop_image(inverted_img)
        height, width = cropped_char.shape
        if height > min_height:
            chars.append(cropped_char)

    return chars
    
def process_image(image):
    pil_image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(pil_image)
    image = image.unsqueeze(0)
    
    return image

def predict_image(image, model, device='cpu'):
    image = process_image(image)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        
        _, predicted_idx = torch.max(output, 1)
    
    predicted_class = classes[predicted_idx.item()]
    return predicted_class

if __name__ == '__main__':
    net = GrayscaleNet()
    net.load_state_dict(torch.load('./grayscale_30class_model.pth'))

    processed_img = preprocess_image('image.png')
    column_words = segment_cols(processed_img)

    all_words = []
    for col in column_words:
        words = segment_words(col)
        all_words.extend(words)

    for word_index, word_img in enumerate(all_words):
        print(f"\n--- Word {word_index + 1} ---")
        chars = segment_characters(word_img)
        for char_index, char in enumerate(chars):
            predicted_class = predict_image(char, net)
            print(f'Char {char_index + 1} of word {word_index + 1}: {predicted_class}')
        
            cv2.imshow(f'Word {word_index + 1} - Char {char_index + 1}: {predicted_class}', char)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    
    

