# GS-ProCams: Gaussian Splatting-based Projector-Camera Systems

**[Project Page](https://realqingyue.github.io/GS-ProCams/)  |  [Data](https://westlakeu-my.sharepoint.com/:f:/g/personal/dengqingyue_westlake_edu_cn/EojpBYrI3HFMicy9ggZ6zv8BHlG8AlV26FZYQH3vqdBjUA?e=KEKl4x)**

Official implementation of "GS-ProCams: Gaussian Splatting-based Projector-Camera Systems" [IEEE TVCG/ISMAR'25].

## Quick Start

### Requirements

- CUDA-capable GPU
- C++ Compiler for PyTorch extensions - We used MSVC (Visual Studio 2019) on Windows
- CUDA Toolkit 11.8

### Setup

```bash
# 1. Download
git clone https://github.com/RealQingyue/GS-ProCams.git

# 2. Create conda environment
conda env create -f environment.yml
conda activate gs-procams

# 3. Build submodules
pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn
```

### Run Pipelines

We provide three ready-to-run scripts to reproduce the experiments described in the paper. Before running them, please download the datasets and place them in the "data" directory: our [real-world dataset](https://westlakeu-my.sharepoint.com/:f:/g/personal/dengqingyue_westlake_edu_cn/EojpBYrI3HFMicy9ggZ6zv8BHlG8AlV26FZYQH3vqdBjUA?e=KEKl4x) and the [synthetic dataset](https://github.com/yoterel/nepmap/tree/master) provided by Erel et al.

```bash
# 1. ProCams simulation on the synthetic dataset
bash ./scripts/synthetic.sh

# 2. ProCams simulation on our real-world dataset 
bash ./scripts/real-world.sh

# 3. Compensation pipeline
bash ./scripts/compensate.sh

# We also provide .ps1 scripts for Windows PowerShell. If you are using Windows, please run these scripts instead.
```

## Apply GS-ProCams to your own setup

A small example setup is provided in our [real-world dataset](https://westlakeu-my.sharepoint.com/:f:/g/personal/dengqingyue_westlake_edu_cn/EojpBYrI3HFMicy9ggZ6zv8BHlG8AlV26FZYQH3vqdBjUA?e=KEKl4x). Use the following directory structure:

```text
<data/compensation>
├── patterns
│   ├── calib
│   ├── ref
│   ├── test
│   └── train
└── setups
    └── example
        └── views
            ├── 01
            │   └── cam
            │       └── raw
            │           ├── calib
            │           ├── ref
            │           ├── test
            │           └── train
            ├── 02
            │   └── cam
            │       └── raw
            │           ├── calib
            │           ├── ref
            │           ├── test
            │           └── train
            └── ...
```

We provide a script that runs calibration using COLMAP. Install COLMAP locally and run:

```bash
python scripts/registrate.py -r 'data/compensation' -s example 
```

Train a model for this setup by running:

```bash
python train.py -r 'data/compensation' -s example -m 'output/example'
```

To register a novel viewpoint, capture one photo from that viewpoint while projecting the same calibration pattern (e.g., `patterns/calib/calib.png`) — see `.../cam/raw/calib` in the example — then run:

```bash
python scripts/registrate.py -r 'data/compensation' -s example --view_id 26
```

Simulate novel projections for the newly registered viewpoint:

```bash
python render.py -r 'data/compensation' -s example -m 'output/example' -o 'output/example/render' --views 26
```

The desired image used for compensation is provided at `example/views/26/cam/desired/test`. Generate the compensated projector input with:

```bash
python compensate.py -r 'data/compensation' -s example -m 'output/example' --view_id 26
```

With projector compensation in hand, you can implement additional applications. To create different desired appearances, provide or edit images in `views/26/cam/raw/ref` (for example, using an image editor or texture tools). These images are used to synthesize the desired target for compensation.

## Acknowledgments

This work builds upon [3DGS][1] and [2DGS][2]. We integrate [DiffMorpher][3], [CDC][4], and [LAMA][5] with our method to achieve various applications. [DeProCams][6] provides inspiration for differentiable ProCams. We thank [Nepmap][7] for providing the synthetic dataset and the anonymous reviewers for their valuable feedback.


## License

This software is available free of charge for non-commercial, non-profit use and may be redistributed under the terms specified in [license](LICENSE). 

**Note**: This work incorporates components from [3DGS][1], [2DGS][2], and other third-party projects. Please ensure compliance with their respective licenses.

[1]: https://github.com/graphdeco-inria/gaussian-splatting
[2]: https://github.com/hbb1/2d-gaussian-splatting
[3]: https://github.com/Kevin-thu/DiffMorpher
[4]: https://github.com/roy-hachnochi/cross-domain-compositing
[5]: https://github.com/advimman/lama
[6]: https://github.com/BingyaoHuang/DeProCams
[7]: https://github.com/yoterel/nepmap/tree/master

## Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{Deng2025GS-ProCams,
  author={Deng, Qingyue and Li, Jijiang and Ling, Haibin and Huang, Bingyao},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={GS-ProCams: Gaussian Splatting-Based Projector-Camera Systems}, 
  year={2025},
  doi={10.1109/TVCG.2025.3616841}
}
```