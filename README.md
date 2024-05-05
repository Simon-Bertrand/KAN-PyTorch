# Python library : torch_kan

Simple implementation of a KAN Layer following the below paper. We use an optimized version of B-Splines computation.
Only grid extension feature is implemented, where you are able to arbitrary increase the number of control points to get less smooth splines curves.
We try the simple MNIST classification task using KAN in the example/sb folder.
We are open for PR concerning library improvements, adding missing features such as automatic pruning and fixing symbolics and correcting wrong interpretation of the original paper.
<br />
<br />
<br />
In this library, the paper from Ziming Liu et al. has been implemented :
<br />


# References :

- Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024, April 30). KAN: Kolmogorov-Arnold Networks. arXiv.org. https://arxiv.org/abs/2404.19756v1


<hr />


# Install library



```bash
pip install https://github.com/Simon-Bertrand/KAN-PyTorch/archive/main.zip

```

# Import library



```python
import torch_kan
```

