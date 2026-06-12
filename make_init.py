import torch
from cldm.model import create_model

# Tumhara model (jisme custom modules hain)
model = create_model('./models/cldm_v15.yaml')

# Pretrained ControlNet weights
pretrained = torch.load('./models/control_sd15_canny.pth', map_location='cpu')
if 'state_dict' in pretrained:
    pretrained = pretrained['state_dict']

# strict=False → pretrained weights load honge, custom modules apne init pe rahenge
missing, unexpected = model.load_state_dict(pretrained, strict=False)

print("Missing keys (tumhare naye modules - inhe train karna hai):", len(missing))
print("Unexpected keys (pretrained me extra - 0 hona chahiye):", len(unexpected))
print("\nSample missing keys:")
for k in missing[:8]:
    print("  ", k)

# Naya init checkpoint save karo
torch.save(model.state_dict(), './models/control_canny_init.ckpt')
print("\nSaved: ./models/control_canny_init.ckpt")
