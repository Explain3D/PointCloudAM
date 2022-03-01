# Visualizing Global Explanations of Point Cloud DNNs

This work is based on existing studies: the generative model is based on [this repo](https://github.com/cihanongun/Point-Cloud-Autoencoder), the classification model is based on [this repo](https://github.com/charlesq34/pointnet). Please build the environment according to the corresponding requirements.

Usage:
1. Train PointNet to have a classification model first, follow the step of [this repo](https://github.com/charlesq34/pointnet).
2. Train our 3 different kinds of generative model via PointCloudAE.py, AED.py and NAED.py
3. Run quantitative AM process on ModelNet40 for corresponding generative models via vanilla_VAE_AM_batch.py, AED_AM_batch.py and NAED_AM_batch.py
<img src="https://github.com/Explain3D/PointCloudAM/blob/main/pics/visu_example.png" width="200" height="400" />
5. You can evaluate AM examples via different metrics: Chamfer, EMD, FID, modified Inception Score and our Point Cloud Activation Maximization Score via Chanfer_eval.py, EMD_eval.py, FID_eval.py, m_IS_eval.py and AM_eval.py
