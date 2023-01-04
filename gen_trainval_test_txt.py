import os


trainval_path = r'/workspace/fuye/Oriented-Object-Detection/DOTA1_0/test/images'
with open(r'/workspace/fuye/Oriented-Object-Detection/DOTA1_0/test.txt', 'w', encoding='utf-8') as f:
    img_name = os.listdir(trainval_path)
    for img in img_name:
        name = os.path.splitext(img)[0]
        f.write(name+'\n')
