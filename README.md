# GerDetection

## 📌 Introduction
This repository contains the official implementation of  
**"Poverty Mapping in Mongolia with AI-based Ger Detection Reveals Urban Slums Persist After the COVID-19 Pandemic."**

The project leverages a deep learning–based segmentation model to detect *gers* (traditional Mongolian dwellings) from satellite imagery.  
By identifying ger areas, this framework provides insights into urban poverty mapping and helps analyze the persistence of informal settlements in Ulaanbaatar and other regions.

---

## Project Structure
```
GerDetection/
├── model/ # Model definition files
├── augmentation.py # Data augmentation functions
├── dataloader.py # Data loading scripts
├── eval.py # Evaluation scripts
├── main.py # Main training/inference entry point
└── utils.py # Utility functions
```
## Installation
```bash
git clone https://github.com/dlsumn/GerDetection.git
cd GerDetection
```

## Usage
```bash
python main.py --epochs 100 --lr 0.01 --weight 0.4
```
### Arguments
- ```--epochs```: number of training epochs
- ```--weight```: class weight for the non-slum category (used in Focal Loss)
- ```--lr```: learning rate

## Requirements
- Python >= 3.7
- PyTorch >= l.8.1
- CUDA >= 11.1
