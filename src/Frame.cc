
#include "Frame.h"
#include "SPextractor.h"
#include <thread>

#include <chrono>

namespace ORB_SLAM2
{

    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame()
    {
    }

    // Copy Constructor
    Frame::Frame(const Frame &frame)
        : mExtractor(frame.mExtractor),
          mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
          N(frame.N), mvKeys(frame.mvKeys), mvKeysUn(frame.mvKeysUn),
          mDescriptors(frame.mDescriptors.clone()), mvbOutlier(frame.mvbOutlier),
          mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
          mfLogScaleFactor(frame.mfLogScaleFactor), mvScaleFactors(frame.mvScaleFactors),
          mvInvScaleFactors(frame.mvInvScaleFactors), mvLevelSigma2(frame.mvLevelSigma2),
          mvInvLevelSigma2(frame.mvInvLevelSigma2), mvKeysClasses(frame.mvKeysClasses),
          mvKeysSemanticRegions(frame.mvKeysSemanticRegions)
    {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];
    }

    // Monocular with Semantic
    Frame::Frame(const cv::Mat &img, const cv::Mat &sem,
                 SPextractor *extractor, cv::Mat &K,
                 cv::Mat &distCoef)
        : mExtractor(extractor), mK(K.clone()),
          mDistCoef(distCoef.clone())
    {

        // SetGTPose(mGT_T);

        // Scale Level Info
        mnScaleLevels = extractor->GetLevels();
        mfScaleFactor = extractor->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = extractor->GetScaleFactors();
        mvInvScaleFactors = extractor->GetInverseScaleFactors();
        mvLevelSigma2 = extractor->GetScaleSigmaSquares();
        mvInvLevelSigma2 = extractor->GetInverseScaleSigmaSquares();

        // ORB extraction
        int i = 0;
        do
        {
            ExtractFeature(img);
            N = mvKeys.size();
            i++;
        } while (N < 2 && i < 10);

        // std::cout<<"N: " << N << std::endl;

        ushort numberOfClasses = 19;

        mvKeysClasses.reserve(N);
        mvKeysSemanticRegions.reserve(N);

        cv::Mat IDs_CV8U, circle;
        std::vector<uchar> circle_vec;
        sem.convertTo(IDs_CV8U, CV_8U);

        for (int i = 0; i < N; i++)
        {

            const cv::KeyPoint &kp = mvKeys[i];

            int class_id = sem.at<int>(static_cast<int>(kp.pt.y), static_cast<int>(kp.pt.x));
            mvKeysClasses.push_back(class_id);

            std::vector<uchar> patch;
            cv::Mat regionIDHist = cv::Mat::zeros(1, numberOfClasses, CV_32F);

            circle = cv::Mat::zeros(kp.size, kp.size, CV_8U);
            cv::circle(circle, cv::Point(int(kp.size / 2), int(kp.size / 2)), static_cast<int>(kp.size / 2), 1, cv::FILLED);
            circle_vec.assign(circle.datastart, circle.dataend);

            cv::Mat region = IDs_CV8U(cv::Range(int(kp.pt.y - kp.size / 2), int(kp.pt.y + kp.size / 2)),
                                      cv::Range(int(kp.pt.x - kp.size / 2), int(kp.pt.x + kp.size / 2)))
                                 .clone();

            patch.assign(region.datastart, region.dataend);

            float totalPixels = 0;
            for (int jjj = 0; jjj < kp.size * kp.size; jjj++)
            {
                if (circle_vec[jjj])
                {
                    ++regionIDHist.at<float>(patch[jjj]);
                    ++totalPixels;
                }
            }

            regionIDHist = ((regionIDHist / (totalPixels)) >= 0.1);
            regionIDHist.convertTo(regionIDHist, CV_8U);
            mvKeysSemanticRegions.push_back(regionIDHist);
        }

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        mvbOutlier = std::vector<bool>(N, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations)
        {
            ComputeImageBounds(img);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        AssignFeaturesToGrid();
    }

    void Frame::AssignFeaturesToGrid()
    {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++)
        {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractFeature(const cv::Mat &im)
    {
        (*mExtractor)(im, cv::Mat(), mvKeys, mDescriptors);
    }

    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
    {
        std::vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = std::max(0, (int)std::floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)std::ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = std::max(0, (int)std::floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)std::ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
        {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
            {
                const std::vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels)
                    {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = std::round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = std::round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        // Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void Frame::UndistortKeyPoints()
    {
        if (mDistCoef.at<float>(0) == 0.0)
        {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++)
        {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++)
        {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft)
    {
        if (mDistCoef.at<float>(0) != 0.0)
        {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
        }
        else
        {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

} // namespace ORB_SLAM
