import os

VIDEO_DIR = './data/videos'

files = [f for f in os.listdir(VIDEO_DIR)]

for f in files:
    ext = f.split('.')[-1]
    prefix = f.split('[')[0].strip().replace(' ', '_')
    num = 1
    dest_name = prefix + "_" + str(num).zfill(2)
    while any(x.startswith(dest_name) for x in os.listdir(VIDEO_DIR)):
        num += 1
        dest_name = prefix + "_" + str(num).zfill(2)

    src = os.path.join(VIDEO_DIR, f)
    dest = os.path.join(VIDEO_DIR, dest_name + "." + ext)

    print(src, "->", dest)
    os.rename(src, dest)

