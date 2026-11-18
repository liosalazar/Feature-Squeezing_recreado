from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from torchvision.transforms.functional import to_pil_image

BASE = Path(__file__).resolve().parents[1]
ASSETS = BASE / 'data'
OUT = BASE / 'results'
(BASE / '.model_cache').mkdir(exist_ok=True)
torch.hub.set_dir(str(BASE / '.model_cache'))
OUT.mkdir(exist_ok=True)
weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
mean = torch.tensor(weights.transforms().mean, device=device).view(1,3,1,1)
std = torch.tensor(weights.transforms().std, device=device).view(1,3,1,1)
categories = weights.meta['categories']

@torch.no_grad()
def predict(x):
    logits = model((x-mean)/std)
    probs = logits.softmax(1)
    vals, idx = probs.max(1)
    return [(int(i), categories[int(i)], float(v)) for i,v in zip(idx, vals)]

def mifgsm(x, y, eps=5/255, steps=10, decay=1.0):
    alpha=eps/steps
    adv=x.detach().clone()
    momentum=torch.zeros_like(adv)
    for _ in range(steps):
        adv.requires_grad_(True)
        loss=torch.nn.functional.cross_entropy(model((adv-mean)/std), y)
        grad=torch.autograd.grad(loss, adv)[0]
        denom=grad.abs().mean(dim=(1,2,3),keepdim=True).clamp_min(1e-12)
        momentum=decay*momentum+grad/denom
        adv=(adv.detach()+alpha*momentum.sign()).clamp(0,1)
        adv=torch.max(torch.min(adv,x+eps),x-eps)
    return adv.detach()

records=[]
for label, cls, source in [
    ('school_bus','school bus',ASSETS/'school_bus_source.png'),
    ('toaster','toaster',ASSETS/'toaster_source.png'),
]:
    image=Image.open(source).convert('RGB')
    x=Compose([Resize(232), CenterCrop(224), ToTensor()])(image).unsqueeze(0).to(device)
    if label == 'school_bus': to_pil_image(x[0].cpu()).save(OUT/'school_bus_clean_224.png')
    y=torch.tensor([categories.index(cls)],device=device)
    before=predict(x)[0]
    adv=mifgsm(x,y)
    after=predict(adv)[0]
    l_inf=float((adv-x).abs().max().item())
    Image.fromarray(np.round(adv.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(OUT/f'{label}_mifgsm.png')
    records.append({'image':label,'ground_truth':cls,'clean_prediction':before[1],'clean_top1_confidence':before[2],
                    'attack':'MI-FGSM','epsilon':5/255,'steps':10,'adversarial_prediction':after[1],
                    'adversarial_top1_confidence':after[2],'clean_correct':before[1]==cls,'attack_success':(before[1]==cls and after[1]!=cls),'measured_linf':l_inf})
    print(label,records[-1])
(OUT/'attack_results.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')




