import os
import glob
import time
import shutil


t0 = time.time()

paths = []
for path in sorted(glob.glob(os.path.join("MNIST_like_", "*")), key=lambda x: int(x.split("_")[-1])):
    paths.append(path)

for i, path in enumerate(paths):
    new_path = os.path.join(os.path.dirname(path), str(i))
    shutil.move(path, new_path) 



print("Done in {:.1f} s.".format(time.time() - t0))

