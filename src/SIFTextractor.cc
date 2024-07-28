#include "SIFTextractor.h"

namespace ORB_SLAM2
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    static float IC_Angle(const Mat &image, Point2f pt, const vector<int> &u_max)
    {
        int m_01 = 0, m_10 = 0;

        const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v * step], val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return fastAtan2((float)m_01, (float)m_10);
    }

    SIFTextractor::SIFTextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                                 int _iniThFAST, int _minThFAST) : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
                                                                   iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;
        for (int i = 1; i < nlevels; i++)
        {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for (int i = 0; i < nlevels; i++)
        {
            mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
            mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for (int level = 0; level < nlevels - 1; level++)
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

        // This is for orientation
        //  pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
    }

    static void computeOrientation(const Mat &image, vector<KeyPoint> &keypoints, const vector<int> &umax)
    {
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                        keypointEnd = keypoints.end();
             keypoint != keypointEnd; ++keypoint)
        {
            keypoint->angle = IC_Angle(image, keypoint->pt, umax);
        }
    }

    vector<cv::KeyPoint> SIFTextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys, const int &minX,
                                                          const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

        const float hX = static_cast<float>(maxX - minX) / nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode *> vpIniNodes;
        vpIniNodes.resize(nIni);

        for (int i = 0; i < nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
            ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
            ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
            ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        // Associate points to childs
        for (size_t i = 0; i < vToDistributeKeys.size(); i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while (lit != lNodes.end())
        {
            if (lit->vKeys.size() == 1)
            {
                lit->bNoMore = true;
                lit++;
            }
            else if (lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int, ExtractorNode *>> vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size() * 4);

        while (!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while (lit != lNodes.end())
            {
                if (lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1, n2, n3, n4;
                    lit->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0)
                    {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0)
                    {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0)
                    {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0)
                    {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit = lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
            {
                bFinish = true;
            }
            else if (((int)lNodes.size() + nToExpand * 3) > N)
            {

                while (!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int, ExtractorNode *>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                    for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--)
                    {
                        ExtractorNode n1, n2, n3, n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                        // Add childs if they contain points
                        if (n1.vKeys.size() > 0)
                        {
                            lNodes.push_front(n1);
                            if (n1.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n2.vKeys.size() > 0)
                        {
                            lNodes.push_front(n2);
                            if (n2.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n3.vKeys.size() > 0)
                        {
                            lNodes.push_front(n3);
                            if (n3.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if (n4.vKeys.size() > 0)
                        {
                            lNodes.push_front(n4);
                            if (n4.vKeys.size() > 1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if ((int)lNodes.size() >= N)
                            break;
                    }

                    if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                        bFinish = true;
                }
            }
        }

        // Retain the best point in each node
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++)
        {
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint *pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;

            for (size_t k = 1; k < vNodeKeys.size(); k++)
            {
                if (vNodeKeys[k].response > maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }

    void SIFTextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>> &allKeypoints)
    {
        allKeypoints.resize(nlevels);

        const float W = 30;

        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD - 3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
            const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures * 10);

            const float width = (maxBorderX - minBorderX);
            const float height = (maxBorderY - minBorderY);

            const int nCols = width / W;
            const int nRows = height / W;
            const int wCell = ceil(width / nCols);
            const int hCell = ceil(height / nRows);

            for (int i = 0; i < nRows; i++)
            {
                const float iniY = minBorderY + i * hCell;
                float maxY = iniY + hCell + 6;

                if (iniY >= maxBorderY - 3)
                    continue;
                if (maxY > maxBorderY)
                    maxY = maxBorderY;

                for (int j = 0; j < nCols; j++)
                {
                    const float iniX = minBorderX + j * wCell;
                    float maxX = iniX + wCell + 6;
                    if (iniX >= maxBorderX - 6)
                        continue;
                    if (maxX > maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;
                    FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, iniThFAST, true);

                    if (vKeysCell.empty())
                    {
                        FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                             vKeysCell, minThFAST, true);
                    }

                    if (!vKeysCell.empty())
                    {
                        for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++)
                        {
                            (*vit).pt.x += j * wCell;
                            (*vit).pt.y += i * hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }
                }
            }

            vector<KeyPoint> &keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

            const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = keypoints.size();
            for (int i = 0; i < nkps; i++)
            {
                keypoints[i].pt.x += minBorderX;
                keypoints[i].pt.y += minBorderY;
                keypoints[i].octave = level;
                keypoints[i].size = scaledPatchSize;
            }
        }

        // compute orientations
        for (int level = 0; level < nlevels; ++level)
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }

    void SIFTextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint>> &allKeypoints)
    {
        allKeypoints.resize(nlevels);

        float imageRatio = (float)mvImagePyramid[0].cols / mvImagePyramid[0].rows;

        for (int level = 0; level < nlevels; ++level)
        {
            const int nDesiredFeatures = mnFeaturesPerLevel[level];

            const int levelCols = sqrt((float)nDesiredFeatures / (5 * imageRatio));
            const int levelRows = imageRatio * levelCols;

            const int minBorderX = EDGE_THRESHOLD;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
            const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

            const int W = maxBorderX - minBorderX;
            const int H = maxBorderY - minBorderY;
            const int cellW = ceil((float)W / levelCols);
            const int cellH = ceil((float)H / levelRows);

            const int nCells = levelRows * levelCols;
            const int nfeaturesCell = ceil((float)nDesiredFeatures / nCells);

            vector<vector<vector<KeyPoint>>> cellKeyPoints(levelRows, vector<vector<KeyPoint>>(levelCols));

            vector<vector<int>> nToRetain(levelRows, vector<int>(levelCols, 0));
            vector<vector<int>> nTotal(levelRows, vector<int>(levelCols, 0));
            vector<vector<bool>> bNoMore(levelRows, vector<bool>(levelCols, false));
            vector<int> iniXCol(levelCols);
            vector<int> iniYRow(levelRows);
            int nNoMore = 0;
            int nToDistribute = 0;

            float hY = cellH + 6;

            for (int i = 0; i < levelRows; i++)
            {
                const float iniY = minBorderY + i * cellH - 3;
                iniYRow[i] = iniY;

                if (i == levelRows - 1)
                {
                    hY = maxBorderY + 3 - iniY;
                    if (hY <= 0)
                        continue;
                }

                float hX = cellW + 6;

                for (int j = 0; j < levelCols; j++)
                {
                    float iniX;

                    if (i == 0)
                    {
                        iniX = minBorderX + j * cellW - 3;
                        iniXCol[j] = iniX;
                    }
                    else
                    {
                        iniX = iniXCol[j];
                    }

                    if (j == levelCols - 1)
                    {
                        hX = maxBorderX + 3 - iniX;
                        if (hX <= 0)
                            continue;
                    }

                    Mat cellImage = mvImagePyramid[level].rowRange(iniY, iniY + hY).colRange(iniX, iniX + hX);

                    cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

                    FAST(cellImage, cellKeyPoints[i][j], iniThFAST, true);

                    if (cellKeyPoints[i][j].size() <= 3)
                    {
                        cellKeyPoints[i][j].clear();

                        FAST(cellImage, cellKeyPoints[i][j], minThFAST, true);
                    }

                    const int nKeys = cellKeyPoints[i][j].size();
                    nTotal[i][j] = nKeys;

                    if (nKeys > nfeaturesCell)
                    {
                        nToRetain[i][j] = nfeaturesCell;
                        bNoMore[i][j] = false;
                    }
                    else
                    {
                        nToRetain[i][j] = nKeys;
                        nToDistribute += nfeaturesCell - nKeys;
                        bNoMore[i][j] = true;
                        nNoMore++;
                    }
                }
            }

            // Retain by score

            while (nToDistribute > 0 && nNoMore < nCells)
            {
                int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute / (nCells - nNoMore));
                nToDistribute = 0;

                for (int i = 0; i < levelRows; i++)
                {
                    for (int j = 0; j < levelCols; j++)
                    {
                        if (!bNoMore[i][j])
                        {
                            if (nTotal[i][j] > nNewFeaturesCell)
                            {
                                nToRetain[i][j] = nNewFeaturesCell;
                                bNoMore[i][j] = false;
                            }
                            else
                            {
                                nToRetain[i][j] = nTotal[i][j];
                                nToDistribute += nNewFeaturesCell - nTotal[i][j];
                                bNoMore[i][j] = true;
                                nNoMore++;
                            }
                        }
                    }
                }
            }

            vector<KeyPoint> &keypoints = allKeypoints[level];
            keypoints.reserve(nDesiredFeatures * 2);

            const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

            // Retain by score and transform coordinates
            for (int i = 0; i < levelRows; i++)
            {
                for (int j = 0; j < levelCols; j++)
                {
                    vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                    KeyPointsFilter::retainBest(keysCell, nToRetain[i][j]);
                    if ((int)keysCell.size() > nToRetain[i][j])
                        keysCell.resize(nToRetain[i][j]);

                    for (size_t k = 0, kend = keysCell.size(); k < kend; k++)
                    {
                        keysCell[k].pt.x += iniXCol[j];
                        keysCell[k].pt.y += iniYRow[i];
                        keysCell[k].octave = level;
                        keysCell[k].size = scaledPatchSize;
                        keypoints.push_back(keysCell[k]);
                    }
                }
            }

            if ((int)keypoints.size() > nDesiredFeatures)
            {
                KeyPointsFilter::retainBest(keypoints, nDesiredFeatures);
                keypoints.resize(nDesiredFeatures);
            }
        }

        // and compute orientations
        for (int level = 0; level < nlevels; ++level)
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }

    void SIFTextractor::computeDescriptors(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors)
    {
        // extractor.compute(image, keypoints, descriptors );
    }

    void SIFTextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints,
                                   OutputArray _descriptors)
    {
        if (_image.empty())
            return;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);

        // Pre-compute the scale pyramid
        ComputePyramid(image);

        vector<vector<KeyPoint>> allKeypoints;
        ComputeKeyPointsOctTree(allKeypoints);
        // ComputeKeyPointsOld(allKeypoints);

        Mat descriptors;

        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if (nkeypoints == 0)
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        _keypoints.clear();
        _keypoints.reserve(nkeypoints);

        int offset = 0;
        for (int level = 0; level < nlevels; ++level)
        {
            vector<KeyPoint> &keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();

            if (nkeypointsLevel == 0)
                continue;

            // preprocess the resized image
            Mat workingMat = mvImagePyramid[level].clone();
            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

            // Compute the descriptors
            Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
            computeDescriptors(workingMat, keypoints, desc);

            offset += nkeypointsLevel;

            // Scale keypoint coordinates
            if (level != 0)
            {
                float scale = mvScaleFactor[level]; // getScale(level, firstLevel, scaleFactor);
                for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                                keypointEnd = keypoints.end();
                     keypoint != keypointEnd; ++keypoint)
                    keypoint->pt *= scale;
            }
            // And add the keypoints to the output
            _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
        }
    }

    void SIFTextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
            Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if (level != 0)
            {
                resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101 + BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }
    }

}