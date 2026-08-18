# Auto-Turret

### NOTE: 
- This project is officially a project under DSAI club, IIT Bhilai and is **strictly for educational purposes**
- In case other individuals and/or organisations want to use this project as a base template or as a product, or in general, for other than educational purposes, contact DSAI club, IIT Bhilai

---

## Index



---

## Main Idea

<ul>
  <li> Automatic Turret can be divided into 2 categories, namely hardware and software side. </li>
  <li> Hardware can be further broken down into <strong>Mounting stand</strong>, <strong>camera mount</strong>, and <strong>laser mount</strong>. This part will mostly be managed by other relevant clubs and, if possible, will be developed by DSAI club on it's own </li>
  <li> Software can be further broken down into 3 categories:
    <ol>
      <li> Object detection and tracking </li>
      <li> Laser pointing and firing mechanism </li>
      <li> Input </li>
    </ol>
    </li>
</ul>

---

## Installation guide

- First clone the repo
```
git clone https://github.com/StellarYolk/Auto-Turret.git
cd Auto-Turret
```

- Create a virtual environment for the best experience and install the dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Check if you have CUDA toolkit installed by running
```
ls /usr/local | grep cuda
# if this returns successfully, CUDA_HOME=/usr/local/cuda-X.X where X represents the version number

# if that returns empty, then run this to check if CUDA toolkit is installed
nvcc --version

# if that returns valid then run and CUDA_HOME = the path minus the /bin/nvcc
which nvcc
```

If you don't have CUDA toolkit installed, then install with steps relevant to your system

- Clone **GroundingDINO**
```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

export CUDA_HOME=path_to_your_actual_CUDA_install

pip install -e .
```
**NOTE** : This may throw an error based on your installed version of torch and torchvision. 

In that case, follow these steps


<ol>
  <li>Go to the folder <code>GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn</code></li>
  <li>
    In the file <code>ms_deform_attn_cuda.cu</code>
    <ul>
      <li>in the lines 35 and 101 in the assertion check function change <code>value.type().is_cuda()</code> to <code>value.is_cuda()</code></li>
      <li>in the lines 65 aND 135  in the dispatch function change <code>value.type()</code> to <code>value.scalar_type()</code></li>
    </ul>
  </li>
  <li>In the file <code>ms_deform_attn.h</code>, in the lines 30 and 52 change <code>value.type().is_cuda()</code> to <code>value.is_cuda()</code></li>
</ol>



- Installing the pre-trained weight for GroundingDINO
```
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
**NOTE** : This uses a lot of computational power so make sure you have atleast 8 GB of VRAM