import os
import sys
import torch
import pathlib

torch.serialization.add_safe_globals([pathlib.PosixPath])

path = sys.argv[1]
m = torch.load(path, map_location="cpu")
new_name = os.path.splitext(path)[0] + '_state_dict.ckpt'
torch.save(m['state_dict'], new_name)


