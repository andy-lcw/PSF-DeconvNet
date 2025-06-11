# PSF-DeconvNet: Improving the resolution of seismic migration image based on a point spread function deconvolution neural network

## 🔍 Description

**PSF-DeconvNet** is a lightweight and efficient neural network designed for automatic deconvolution of point spread functions (PSFs) in seismic migration imaging. Unlike conventional deblurring approaches that require precomputed deconvolution operators, PSF-DeconvNet learns to predict PSF deconvolution operators in the **wavenumber domain** using a physics-informed self-supervised training strategy. This framework significantly reduces computational and storage costs and improves resolution in seismic migration images.

The network integrates:
- A multi-layer perceptron (MLP)-based encoder-decoder architecture
- Dual self-attention modules
- A physics-guided loss based on matching the deconvolution result to an ideal impulse function

## 🚀 Key Features

- **Self-supervised training** without requiring deconvolution operator labels  
- **Wave-number domain formulation** for stable and interpretable learning  
- **Pixel-wise deblurring** with high efficiency and generalization capability  
- **Support for large-scale PSF datasets** with dynamic interpolation  
- **Broad applicability** to 2D/3D seismic imaging and potentially to other PSF-related inverse problems (e.g., remote sensing, astrophysics, microscopy)

## 📁 Project Structure

PSF-DeconvNet/  
├── data/                # PSF generation & preprocessing scripts  
├── models/              # Network architecture (encoder, decoder, attention)  
├── utils/               # Helper functions (FFT, loss, visualization)  
├── configs/             # YAML config files for experiments  
├── results/             # Visualization and evaluation outputs  
├── train.py             # Training script  
├── test.py              # Evaluation and prediction script  
└── README.md            # Project description  

## 📦 Dependencies

- Python >= 3.8  
- PyTorch >= 1.10  
- NumPy  
- SciPy  
- Matplotlib  
- FFT support (e.g., `torch.fft`)

🛠️ Usage

1. Generate PSFs
   
Prepare a grid of point scatterers and use smooth background velocity models to compute PSFs via demigration and migration.

2. Train PSF-DeconvNet & Predicion

python PSF_DeconvNet.py 11 Marmousi2 

3. Apply to Migration Image

Use the predicted deconvolution operators to enhance the resolution of the seismic migration image.

## 📊 Results

- ✅ **Substantial improvements** in both spatial and spectral resolution  
- 🏆 **Competitive or superior** to traditional LSRTM and CNN-based methods  
- ⚡️ **Faster runtime** (up to **10×** compared with LSRTM and CNN-PSF methods)  
- 🌍 **High generalization** to perturbed velocity models  
  (validated on Marmousi scenario)
 
📈 Citation

If you use this code or model in your research, please cite:
```
@article{liu2025psfdeconvnet,
  title={PSF-DeconvNet: Improving the resolution of seismic migration image based on a point spread function deconvolution neural network},
  author={Cewen Liu and Haohuan Fu},
  journal={Journal of Geophysical Research: Machine Learning and Computation},
  year={2025},
  doi={10.1029/2025JBXXXXXX}
}
```
🔐 License

This project is released under the MIT License.

📬 Contact

Cewen Liu  
Department of Earth System Science  
Tsinghua University  
📧 lcw_17_tsinghua@163.com  
📍 Beijing, China
