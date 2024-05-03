# Python library : torch_cmif

The torch_cmif library provides a fast implementation the cross mutual information between one real image and one another on PyTorch.
<br />
<br />
<br />
In this library, the paper from J. Öfverstedt et al. has been implemented :
<br />


# References :

- Johan Öfverstedt, Joakim Lindblad, Nataša Sladoje, (2022). Fast computation of mutual information in the frequency domain with applications to global multimodal image alignment, - https://www.sciencedirect.com/science/article/pii/S0167865522001817


<hr />


# Install library



```bash
%%bash
if !python -c "import torch_cmif" 2>/dev/null; then
    pip install https://github.com/Simon-Bertrand/FastCMIF-PyTorch/archive/main.zip
fi
```

# Import library



```python
import torch_cmif
```


```python
!pip install -q matplotlib torchvision
import torch
import matplotlib.pyplot as plt
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


## LOAD IMAGE AND TEST IF RANDOM EXTRACTED CENTER POSITIONS ARE CORRECTLY FOUND


Install notebook dependencies



```python
!pip install -q requests
import requests
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


Load Mandrill image



```python
import tempfile
import torchvision
import torch.nn.functional as F

with tempfile.NamedTemporaryFile() as fp:
    fp.write(
        requests.get(
            "https://upload.wikimedia.org/wikipedia/commons/a/ab/Mandrill-k-means.png"
        ).content
    )
    im = F.interpolate(
        (
            torchvision.io.read_image(
                fp.name, torchvision.io.ImageReadMode.RGB
            )
            .unsqueeze(0)
            .to(torch.float64)
            .div(255)
        ),
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )
```

Run multiple tests to check if random crop center is correclty found by the ZNCC.



```python
import random

success = 0
failed = 0
pts = []
for _ in range(16):
    imH = random.randint(64, 128)
    imW = random.randint(64, 128)
    i = random.randint(imH // 2 + 1, im.size(-2) - imH // 2 - 1)
    j = random.randint(imW // 2 + 1, im.size(-1) - imW // 2 - 1)

    imT = im[
        :, :, i - imH // 2 : i + imH // 2 + 1, j - imW // 2 : j + imW // 2 + 1
    ]
    if (
        (
            torch_cmif.FastCMIF.findArgmax(
                torch_cmif.FastCMIF(8, "none")(im, imT)
            )
            - torch.Tensor([[[i]], [[j]]])
        ).abs()
        < 3
    ).all():
        pts += [
            dict(
                i=i,
                imH=imH,
                imW=imW,
                j=j,
                success=True,
            )
        ]
        success += 1
    else:
        pts += [
            dict(
                i=i,
                imH=imH,
                imW=imW,
                j=j,
                success=False,
            )
        ]
        failed += 1

plt.imshow(im[0].moveaxis(0, -1))
ax = plt.gca()
for pt in pts:
    ax.add_patch(
        plt.Rectangle(
            (pt["j"] - pt["imW"] // 2, pt["i"] - pt["imH"] // 2),
            pt["imW"],
            pt["imH"],
            edgecolor="g" if pt["success"] else "r",
            facecolor="none",
            linewidth=0.5,
        )
    )
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](figs/README_14_1.png)
    



```python
ans = torch_cmif.FastCMIF(8, "sum")(im, imT)
plt.imshow(ans[0].mean(0))
plt.title("CMIF")
```




    Text(0.5, 1.0, 'CMIF')




    
![png](figs/README_15_1.png)
    



```python
%timeit torch_cmif.FastCMIF(8, "sum")(im, imT)
```

    182 ms ± 2.81 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


Total errors :



```python
dict(success=success, failed=failed)
```




    {'success': 16, 'failed': 0}




```python

```
