
# Neural Interaction Detection (NID) 
**Project from ML Course - Skoltech 2025**

![Algorithm](figures/algorithm.gif)

This repository implements Neural Interaction Detection (NID), a method to detect statistical interactions from neural network weights, based on the ICLR 2018 paper by Tsang et al. [[PDF]](https://openreview.net/pdf?id=ByOfBggRZ)

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Experiments](#experiments)
- [Plotting](#plotting)
- [Citation](#citation)

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── src/experiments/          # Scripts for experimental setups
├── datasets/                 # Synthetic (with ground truth) and real-world datasets
├── neural_detection/         # Core algorithm implementation
│   └── multilayer_perceptron/
├── src/plots/                # Generated plots and visualization scripts
├── demo.ipynb                # Jupyter notebook with key experiments
├── draw_smth.py              # Plotting utilities
└── requirements.txt          # Python dependencies
```

## Getting Started
Explore `demo.ipynb` to run:
1. **Synthetic Function Experiments**: Validate NID on functions with known interactions.
2. **F11-F12 Tests**: Verify interaction detection on pseudo-IID features.
3. **Correlated Features Analysis**: Test performance on cloned/correlated features.

## Experiments
Preconfigured experiments are located in `src/experiments/`. To reproduce:
Use demo.ipynb 

## Plotting
Visualization scripts are in `draw_smth.py`. Generated plots are saved to `src/plots/`:
```python
python draw_smth.py  
```

## Citation
```bibtex
@article{tsang2017detecting,
  title={Detecting statistical interactions from neural network weights},
  author={Tsang, Michael and Cheng, Dehua and Liu, Yan},
  journal={arXiv preprint arXiv:1705.04977},
  year={2017}
}
```


###### Acknowledgements
This implementation is based on the original work by M. Tsang, D. Cheng, and Y. Liu. Developed for educational purposes in the Skoltech 2025 Machine Learning course.
```
