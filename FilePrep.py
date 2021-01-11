import os
from random import shuffle
from math import floor

bw, rgb = [], []

bw_path = 'C:\\Users\\alexp\\Desktop\\MangaColoring\\Berserk\\bw\\bw'
rgb_path = 'C:\\Users\\alexp\\Desktop\\MangaColoring\\Berserk\\rgb\\rgb'


# clean filenames

os.chdir(bw_path)
for f in os.listdir():
    f = f.lower()
    s = f.split('_')
    if s[0] == 'berserk':
        s.pop(0)
    
    f_new = '_'.join(s).replace('-', '_')
    os.rename(f, f_new)

os.chdir(rgb_path)
for f in os.listdir():
    f = f.lower()
    s = f.split('_')
    if s[0] == 'berserk':
        s.pop(0)
    
    f_new = '_'.join(s).replace('-', '_')
    os.rename(f, f_new)


# split bw images into train/test
# split rgb so that bw/rgb pairs are consistent

os.chdir(bw_path)
for f in os.listdir():
    bw.append(f)

os.chdir(rgb_path)
for f in os.listdir():
    rgb.append(f)

shuffle(bw)
idx = floor(len(bw)/5)
bw_test = bw[:idx]

bw_test_no_ext = [s.split('.')[0] for s in bw_test]
rgb_test = []
for b_no_ext in bw_test_no_ext:
    for r in os.listdir():
            r_no_ext = r.split('.')[0]
            if r_no_ext == '_'.join(['c', b_no_ext]):
                rgb_test.append(r)
                break

os.chdir(bw_path)

for f in bw_test:
    new_path = os.path.join("..\\bw_test", f)
    os.rename(f, new_path)

os.chdir(rgb_path)

missing = []

for f in rgb_test:
    new_path = os.path.join("..\\rgb_test", f)
    try:
        os.rename(f, new_path)
    except:
        missing.append(f)

print("Missing rgb files:")
for f in missing:
    print(f)

print(f'No. BW Test Images: {len(bw_test)}')
print(f'No. RGB Test Images: {len(rgb_test)}')