#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include "ExractorNode.h"

#include <torch/torch.h>
#include <torch/script.h>
#include "SuperPoint.h"

namespace ORB_SLAM2
{

    class SPextractor
    {
    public:
        enum
        {
            HARRIS_SCORE = 0,
            FAST_SCORE = 1
        };

        SPextractor(int nfeatures, float scaleFactor, int nlevels,
                    float iniThFAST, float minThFAST);

        ~SPextractor() {}

        // Compute the SP features and descriptors on an image.
        // SP are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors);

        int inline GetLevels()
        {
            return nlevels;
        }

        float inline GetScaleFactor()
        {
            return scaleFactor;
        }

        std::vector<float> inline GetScaleFactors()
        {
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors()
        {
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares()
        {
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares()
        {
            return mvInvLevelSigma2;
        }

        std::vector<cv::Mat> mvImagePyramid;

    protected:
        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints, cv::Mat &_desc);
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                    const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        int nfeatures;
        float scaleFactor;
        int nlevels;
        float iniThFAST;
        float minThFAST;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        std::shared_ptr<torch::jit::script::Module> model;
    };

} // namespace ORB_SLAM

#endif
