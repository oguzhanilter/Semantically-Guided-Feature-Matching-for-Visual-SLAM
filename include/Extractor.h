#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include "ExractorNode.h"

#include <vector>
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace ORB_SLAM2
{

    class Extractor
    {
    public:
        enum
        {
            HARRIS_SCORE = 0,
            FAST_SCORE = 1
        };

        Extractor() = default;
        Extractor(int nfeatures, float scaleFactor, int nlevels,
                  int iniThFAST, int minThFAST);

        ~Extractor() = default;

        virtual void operator()(cv::InputArray image, cv::InputArray mask,
                                std::vector<cv::KeyPoint> &keypoints,
                                cv::OutputArray descriptors) = 0;

        virtual int inline GetLevels() = 0;

        virtual float inline GetScaleFactor() = 0;

        virtual std::vector<float> inline GetScaleFactors() = 0;

        virtual std::vector<float> inline GetInverseScaleFactors() = 0;

        virtual std::vector<float> inline GetScaleSigmaSquares() = 0;

        virtual std::vector<float> inline GetInverseScaleSigmaSquares() = 0;
    };

}

#endif
