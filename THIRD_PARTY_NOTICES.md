# Third-party notices

## Public chest X-rays and modified maps

Source: Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, version 2. DOI: 10.17632/rscbjbr9sj.2. Only the chest X-ray subset is used.

Source and license record: https://data.mendeley.com/datasets/rscbjbr9sj/2

License: Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

The public images in `docs/examples/`, the comparison in `docs/sanity/`, and the recorded assets in `web/demo/` derive from this dataset. Resizing, Grad-CAM maps, colored overlays, and comparison layouts are modifications. Original filenames, selection rules, source links and checkpoint provenance are recorded in `docs/examples.json`, `docs/sanity/results.json`, and `web/demo/manifest.json`.

This project is not endorsed by the dataset authors. No private patient records are included.

## Methods and software

- Grad-CAM: Selvaraju et al., https://arxiv.org/abs/1610.02391
- Parameter-randomization inspiration: Adebayo et al., https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- ResNet-18 implementation and ImageNet weights: torchvision, https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

Python packages retain their respective licenses. Trained weights, the full source dataset and third-party package code are not bundled in the source distribution. This notice documents third-party materials; it does not assign a license to the project's original code.
