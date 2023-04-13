# OIQA_FR_Metrics

Pytorch Implement of Classic Full-Reference Quality Assessment Methods for 360° content


## Instance
```
model = FR_Metric()

>>>
Support models:
['MSE', 'PSNR', 'SSIM', 'WS_PSNR', 'CPP_PSNR', 'S_PSNR', 'S_SSIM'] 
```
      
## Select models
```
model_list = ['PSNR', 'SSIM', 'WS_PSNR', 'CPP_PSNR', 'S_PSNR', 'S_SSIM']                
```

## Runing
```
scores = model(model_list, (ref, dis))

>>>
Generating weight map
saving to ./weight_map/(2048, 4096)_WS_PSNR.pth
Generating weight map
saving to ./weight_map/(2048, 4096)_CPP_PSNR.pth
Generating coordinate
saving to ./weight_map/(2048, 4096)_S_PSNR.pth
Generating patch coordinate
saving to ./weight_map/(1024, 2048)_Stride-10_S_SSIM.pth
```

## Check scores
```
for key in scores:
        print(key, scores[key])

>>>
PSNR tensor([30.7618, ...])
SSIM tensor([0.8725, ...])
WS_PSNR tensor([29.8037, ...])
CPP_PSNR tensor([29.8745, ...])
S_PSNR tensor([29.7890, ...])
S_SSIM tensor([0.8644, ...])
```
