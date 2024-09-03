#ifndef SIFTEXTRACTOR_H
#define SIFTEXTRACTOR_H

#include "Extractor.h"

namespace ORB_SLAM2
{

    class SIFTextractor : public Extractor
    {
    public:
        SIFTextractor() = default;
        SIFTextractor(int nfeatures, float scaleFactor, int nlevels,
                      int iniThFAST, int minThFAST);

        ~SIFTextractor() {}

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
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
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
        std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                    const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
        std::vector<cv::Point> pattern;

        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        std::vector<cv::Mat> mvImagePyramid;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        void computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    };

}

#endif
