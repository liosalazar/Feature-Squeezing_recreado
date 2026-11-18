from pathlib import Path
import json, time, csv
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import to_tensor, to_pil_image

BASE=Path(__file__).resolve().parents[1]
ASSETS=BASE/'data'
OUT=BASE/'results'
OUT.mkdir(exist_ok=True)
# Load the SwinIR architecture exactly from the public author notebook.
source=json.load(open(BASE/'third_party'/'SwinSR-Defense_Autores.ipynb',encoding='utf-8'))['cells'][1]['source']
namespace={}
exec(compile(''.join(source),'author_notebook_cell_1','exec'),namespace)
SwinIR=namespace['SwinIR']

clean=Image.open(OUT/'school_bus_clean_224.png').convert('RGB')
adv=Image.open(OUT/'school_bus_mifgsm.png').convert('RGB')
x=to_tensor(adv).unsqueeze(0)
y=to_tensor(clean).unsqueeze(0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=SwinIR(upscale=1,img_size=(224,224),window_size=8,img_range=1.0,
             depths=[4,4,4],embed_dim=60,num_heads=[4,4,4],mlp_ratio=2,
             upsampler='',drop_rate=0.0,attn_drop_rate=0.0,drop_path_rate=0.0).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.05)
criterion=nn.MSELoss()
x=x.to(device); y=y.to(device)
model.train()
losses=[]
start=time.perf_counter()
for epoch in range(35):
    optimizer.zero_grad(set_to_none=True)
    pred=model(x)
    loss=criterion(pred,y)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    if epoch in (0,4,9,19,34):
        print(f'epoch={epoch+1:02d}/35 mse={losses[-1]:.7f}')
model.eval()
with torch.inference_mode():
    restored=model(x).clamp(0,1)
seconds=time.perf_counter()-start
restored_img=to_pil_image(restored[0].cpu())
restored_img.save(OUT/'school_bus_mifgsm_swinir_trained.png')
torch.save(model.state_dict(),OUT/'swinir_pilot_1pair.pth')
with open(OUT/'training_loss.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['epoch','mse']); w.writerows(enumerate(losses,1))

weights=ResNet50_Weights.IMAGENET1K_V2
(BASE/'.model_cache').mkdir(exist_ok=True); torch.hub.set_dir(str(BASE/'.model_cache'))
classifier=resnet50(weights=weights).eval().to(device)
mean=torch.tensor(weights.transforms().mean,device=device).view(1,3,1,1)
std=torch.tensor(weights.transforms().std,device=device).view(1,3,1,1)
cat=weights.meta['categories']
@torch.no_grad()
def pred(pil):
    # Input images are already transformed to the exact 224x224 attack tensor.
    z=to_tensor(pil).unsqueeze(0).to(device)
    scores=classifier((z-mean)/std).softmax(1)[0]
    i=int(scores.argmax())
    return cat[i],float(scores[i])

results={'image':'school bus','attack':'MI-FGSM','epsilon':'5/255','steps':10,
         'training_pairs':1,'epochs':35,'batch_size':1,'optimizer':'AdamW',
         'learning_rate':1e-4,'weight_decay':0.05,'loss':'MSE',
         'mse_first_epoch':losses[0],'mse_last_epoch':losses[-1],
         'training_seconds':round(seconds,2),
         'pred_clean':pred(clean),'pred_adversarial':pred(adv),'pred_after_pilot_training':pred(restored_img),
         'warning':'Single-pair in-sample demonstration; not a robust-accuracy estimate or full paper reproduction.'}
(OUT/'training_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,indent=2,ensure_ascii=False))
print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

