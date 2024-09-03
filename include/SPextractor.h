#ifndef SPEXTRACTOR_H
#define SPEXTRACTOR_H

#include "Extractor.h"

#include <torch/torch.h>
#include <torch/script.h>
#include "SuperPoint.h"

namespace ORB_SLAM2
{

    class SPextractor : public Extractor
    {
    public:

        SPextractor() = default;
        SPextractor(int nfeatures, float scaleFactor, int nlevels,
                    float iniThFAST, float minThFAST);

        ~SPextractor() {}

        // Compute the SP features and descriptors on an image.
        // SP are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        void operator()(cv::InputArray image, cv::InputArray mask,
                        std::vector<cv::KeyPoint> &keypoints,
                        cv::OutputArray descriptors) override;

        int inline GetLevels() override
        {
            return nlevels;
        }

        float inline GetScaleFactor() override
        {
            return scaleFactor;
        }

        std::vector<float> inline GetScaleFactors() override
        {
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors() override
        {
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares() override
        {
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares() override
        {
            return mvInvLevelSigma2;
        }

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

        std::vector<cv::Mat> mvImagePyramid;

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
