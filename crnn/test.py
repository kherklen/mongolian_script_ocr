import torch
from torchvision import transforms
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

alphabet = "?᠀᠁᠂᠃᠄᠅᠋᠌᠍᠐᠑᠒᠓᠔᠕᠖᠗᠘᠙ᠠᠡᠢᠣᠤᠥᠦᠧᠨᠩᠪᠫᠬᠭᠮᠯᠰᠱᠲᠳᠴᠵᠶᠷᠸᠹᠺᠻᠼᠽᠾ《》 "
char2idx = {c: i + 1 for i, c in enumerate(alphabet)}
char2idx["<BLANK>"] = 0
idx2char = {i: c for c, i in char2idx.items()}

class RescaleHeight:
    def __init__(self, img_height):
        self.img_height = img_height

    def __call__(self, img):
        h = self.img_height
        w = int(img.width * (h / img.height))
        return transforms.functional.resize(img, (h, w))


bichig_to_cyrillic = {
    '᠐':'0', '᠑':'1', '᠒':'2','᠓':'3', '᠔':'4', '᠕':'5',
    '᠖':'6', '᠗':'7', '᠘':'8', '᠙':'9', 'ᠠ':'а', 'ᠡ':'э',
    'ᠢ':'и', 'ᠣ':'о', 'ᠤ':'у', 'ᠥ':'ү', 'ᠦ':'ө', 'ᠧ':'я',
    'ᠨ':'н', 'ᠩ':'и', 'ᠪ':'б', 'ᠫ':'п', 'ᠬ':'х', 'ᠭ':'г',
    'ᠮ':'м', 'ᠯ':'л', 'ᠰ':'с', 'ᠱ':'ш', 'ᠲ':'д', 'ᠳ':'т', 'ᠴ':'ч',
    'ᠵ':'ж', 'ᠶ':'я', 'ᠷ':'р', 'ᠸ':'я', 'ᠹ':'ф', 'ᠺ':'х', 'ᠻ':'к',
    'ᠼ':'ц', 'ᠽ':'з', 'ᠾ':'з', '《':'<','》':'>'
}

def transliterate_to_cyrillic(text):
    return ''.join(bichig_to_cyrillic.get(c, c) for c in text)


image_paths = [
    "32px.png", "normal.png", "normal3.png", "normal4.png", "normal5.png",
    "normal6.png", "normal7.png", "normal8.png", "normal9.png", "normal10.png"
]


model = CRNN(nclass=len(char2idx)).to(device)
model.load_state_dict(torch.load("crnn_model.pth", map_location=device))

for i, image_path in enumerate(image_paths, start=1):
    predicted = predict_single_image(model, image_path, char2idx, idx2char, device)
    translated = transliterate_to_cyrillic(predicted)

    print(f"Image {i}:")
    print("Predicted (Traditional Mongolian):", predicted)
    print("Transliterated (Cyrillic):", translated, "\n")