# ME: Trigger Element Combination Backdoor Attack on Copyright Infringement

Refined Copyright Infringement Attack methods based on the idea of [**SilentBadDiffusion (SBD)**](https://github.com/haonan3/ICML-2024-Oral-SilentBadDiffusion).

[**Paper**](https://arxiv.org/abs/2506.10776)

---

![Attacking process](./assets/attacking_process.png)

## 📖 Overview

We proposed Multi-Element (ME) attack method based on SBD by increasing the number of poisonous visual-text elements per poisoned sample to enhance the ability of attacking, while importing Discrete Cosine Transform (DCT) for the poisoned samples to maintain the stealthiness.

## 🔧 Installation

Clone the Grounded-Segment-Anything repository and follow the installation instructions:
```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

```

## 🔧 Debug (If you really face problem)

1. If you fail to invoke GroundingDINO on GPU like this:

```bash
... Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py:31: UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
  warnings.warn("Failed to load custom C++ ops. Running on CPU mode Only!")
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!

...

Traceback (most recent call last):
... Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py", line 53, in forward
    output = _C.ms_deform_attn_forward(
NameError: name '_C' is not defined
```

You could try to do these:
1) Delete folder 'build' in dir '/Grounded-Segment-Anything/GroundingDINO'
2) Enter the dir '/Grounded-Segment-Anything/GroundingDINO' and do the following **rebuild commands**:

```bash
# re-build & re-intsall
pip uninstall groundingdino
python setup.py build
python setup.py install
```

2. TypeError: BoxAnnotator.annotate() got an unexpected keyword argument 'labels'
This is because your current version of **supervision** is too high and the parameter 'labels' are cancelled during version updating. Return it to version around 0.18 would fix this.

```bash
 pip install supervision==0.18.0
```
