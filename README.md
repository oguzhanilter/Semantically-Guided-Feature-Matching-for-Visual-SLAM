# Semantically-Guided-Feature-Matching-for-Visual-SLAM
Implementation of the ICRA 2024 conference paper: Semantically Guided Feature Matching for Visual SLAM.

![The ORB-SLAM2 pipeline and the components with which our proposed semantic feature matching interacts. Semantic feature descriptors are extracted together with the standard ORB ones. They are then jointly used both in every matching procedures and for generating 3D map points.](resources/SemanticSLAM.png)

# Abstract
We introduce a new algorithm that utilizes semantic information to enhance feature matching in visual SLAM pipelines. The proposed method constructs a high-dimensional semantic descriptor for each detected ORB feature. When integrated with traditional visual ones, these descriptors aid in establishing accurate tentative point correspondences between consecutive frames. Additionally, our semantic descriptors enrich 3D map points, enhancing loop closure detection by providing deeper insights into the underlying map regions. Experiments on public large-scale datasets demonstrate that our technique surpasses the accuracy of established methods. Importantly, given its detector-agnostic nature, our algorithm also amplifies the efficacy of modern keypoint detectors, such as SuperPoint

# Content
This repo contains the example implementation of the feature matching between two frames enhanced by semantic features. This corresponds to the 'map initialization' module of ORB-SLAM2. The rest of the pipeline could not be published due to the ownership of VSO implementation.

# Build using Docker Image
Build the docker image and run the container. Copy the repo inside the container. Please refer to [the official Docker website](https://docs.docker.com/) to see how. 

Alternatively you can use [devcontainer in VSCode](https://code.visualstudio.com/docs/devcontainers/containers) with the provided [dockerfile](dockerfile).

inside the directory 
```console
mkdir build & cd build 
cmake ..
make -j8
```

To run it 
```console
.main <input_data_directory_path>
```

# Input Data
The input directory should contain 2 directory
- images
- semantic_images

_images_ directory should contain the RGB images whereas _semantic_images_ directory should contain images of the semantic class IDs per pixels. 

Please see the example images taken/created from KITTI dataset.

and two txt file
- config.txt
- groundtruth.txt 

Groundtruth should be in TUM format and corresponded the line number with the relative images: 
'''
timestamp x y z q_x q_y q_z q_w 
'''

# Remark
SuperPoint network fails to run on docker image with the following versions although it was running successfully in the server.
- 2.3.1 
- 1.9.1

I will continue working on the repo to make SuperPoint features work.
