# Semantically-Guided-Feature-Matching-for-Visual-SLAM
Implementation of the ICRA 2024 conference paper: Semantically Guided Feature Matching for Visual SLAM.

![The ORB-SLAM2 pipeline and the components with which our proposed semantic feature matching interacts. Semantic feature descriptors are extracted together with the standard ORB ones. They are then jointly used both in every matching procedures and for generating 3D map points.](resources/SemanticSLAM.png)

# Abstract
We introduce a new algorithm that utilizes semantic information to enhance feature matching in visual SLAM pipelines. The proposed method constructs a high-dimensional semantic descriptor for each detected ORB feature. When integrated with traditional visual ones, these descriptors aid in establishing accurate tentative point correspondences between consecutive frames. Additionally, our semantic descriptors enrich 3D map points, enhancing loop closure detection by providing deeper insights into the underlying map regions. Experiments on public large-scale datasets demonstrate that our technique surpasses the accuracy of established methods. Importantly, given its detector-agnostic nature, our algorithm also amplifies the efficacy of modern keypoint detectors, such as SuperPoint

# Content
This repo contains the example implementation of the feature matching between two frames enhanced by semantic features. This corresponds to the 'map initialization' module of ORB-SLAM2. The rest of the pipeline could not be published due to the ownership of VSO implementation.

# Build using Docker Image
Build the docker image and run the container
Copy the repo inside the container
inside directory 
mkdir build & cd build 
cmake ..
make -j8

Alternatively you can use devcontainer in VSCode

# ToDos
- Build Instraction
- Input Explanations
- Refactoring
    - Connection to the paper
- Add paper 