import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
x = torch.randn(1, 3, 14, 14)
with torch.no_grad():
    out = model(x)
print("Output shape 14x14:", out.shape)
x2 = torch.randn(1, 3, 56, 56)
with torch.no_grad():
    out2 = model(x2)
print("Output shape 56x56:", out2.shape)
