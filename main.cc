#include "Frame.h"
#include "Extractor.h"
#include "Initializer.h"
#include "SPextractor.h"
#include "ORBextractor.h"
#include "SIFTextractor.h"

#include "ReadParams.h"

#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#define CV_PI 3.1415926535897932384626433832795

// Feature Extractor Type
ORB = 0 ;
SUPERPOINT = 1;
SIFT = 2;

// Function declearations ---------------------- 
int matcher(ORB_SLAM2::Frame &F1, ORB_SLAM2::Frame &F2, std::vector<int> &vnMatches12, bool withSemantic);
unsigned int compare_semantics(const cv::Mat &m1, const cv::Mat &m2);

cv::Mat loadTxtFile(const std::filesystem::path &filePath);

float rot_error(cv::Mat R_est, cv::Mat R_GT);
float trans_error(cv::Mat t_est, cv::Mat t_GT);
float calculateRMS(std::vector<float> &v);
float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);
void calculateGT(cv::Mat pos_row1, cv::Mat pos_row2, cv::Mat &R_GT, cv::Mat &t_GT);

std::vector<int> linspace(int start_in, int end_in, int num_in);
std::vector<std::filesystem::path> getImageFileNames(const std::filesystem::path &directoryPath);
// --------------------------------------------



// ----- Global variables ---------------------------------------------
// Parameters set inside main
double MAX_DISTANCE_BETWEEN_DESCRIPTORS;
int FEATURE_EXTRACTOR_TYPE; // 0: ORB, 1: SP, 2: SIFT
const float FEATURE_DIST_TRESHOLD ; // feature matching param

// Static Parameters
const int NUMBER_OF_REFERENCES = 250; // 250 images will be selected as static
const std::vector<int> INTERVALS{5, 10, 15}; // for every static image, the 5th, 10th and 15th subsequent images from the static image will be used to calculate position change.  
cv::Mat DISTORTION_COEFFICIENTS = cv::Mat::zeros(4, 1, CV_32F);

// Feature matching related params, taken from ORB-SLAM2
const int HISTO_LENGTH = 30;
const float mfNNratio = 0.7;

// ---------------------------------------------------------------------



int main(int argc, char **argv)
{
    std::filesystem::path dataSetDirectoryPath = argv[1];

    std::cout << "Started! The input directory is " << dataSetDirectoryPath << std::endl;
    std::cout << " **************************************************************" << std::endl;
    std::cout << "     ***************************************************" << std::endl;
    std::cout << "            ***************************************" << std::endl;

    auto params = readParameters(dataSetDirectoryPath / "config.txt");

    // Parameter initilization ----------
    int numberOfInputImages; // will be initilized according to the number of input images

    FEATURE_EXTRACTOR_TYPE = static_cast<int>( params["featureExtractorType"] );

    int nFeatures = static_cast<int>( params["numberOfFeatures"] );
    float fScaleFactor = static_cast<float>( params["fScaleFactor"] );
    int nLevels = static_cast<float>( params["nLevels"] );
    float iniThFAST = static_cast<float>( params["iniThFAST"] );
    float minThFAST = static_cast<float>( params["minThFAST"] );
    float FEATURE_DIST_TRESHOLD = static_cast<float>( params["featureDistThreshold"] ); // 0.30 for SuperPoint, 50 for ORB

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = static_cast<float>( params["intirinsic_fx"] );
    K.at<float>(1,1) = static_cast<float>( params["intirinsic_fy"] );
    K.at<float>(0,2) = static_cast<float>( params["intirinsic_cx"] );
    K.at<float>(1,2) = static_cast<float>( params["intirinsic_cy"] );
    // Parameter initilization done ------

    std::cout << "Parameters are read from the file " << K << std::endl;

    std::cout << "Parameters are read from the file " << dataSetDirectoryPath / "config.txt" << std::endl;

    std::vector<float> rot_error_normal;
    std::vector<float> trans_error_normal;
    std::vector<float> rot_error_sem;
    std::vector<float> trans_error_sem;

    std::vector<std::filesystem::path> vstrImageFilenames;
    std::vector<std::filesystem::path> vstrSemFilenames;

    vstrImageFilenames = getImageFileNames(dataSetDirectoryPath / "images");
    vstrSemFilenames = getImageFileNames(dataSetDirectoryPath / "semantic_images");

    if (vstrImageFilenames.size() == vstrSemFilenames.size())
    {
        numberOfInputImages = vstrImageFilenames.size();
    }
    else
    {
        throw "RGB image count and semantic images count are not equal. Exiting ...";
    }

    if (numberOfInputImages < NUMBER_OF_REFERENCES - INTERVALS.back() )
    {
        throw "Number of images less than the Number of references - largest interval. Please change number of references. Exiting ...";
    }

    std::cout << "Data directory " << dataSetDirectoryPath << " has been set." << std::endl;

    const cv::Mat GTPoses = loadTxtFile(dataSetDirectoryPath / "groundtruth.txt");

    const std::vector<int> indices = linspace(0, numberOfInputImages - INTERVALS.back() - 2, NUMBER_OF_REFERENCES);

    // Set feature extractor
    ORB_SLAM2::Extractor *extractor_ptr;

    if (FEATURE_EXTRACTOR_TYPE == ORB)
    {
        
        extractor_ptr = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, iniThFAST, minThFAST);
        MAX_DISTANCE_BETWEEN_DESCRIPTORS = 256;
    }
    else if (FEATURE_EXTRACTOR_TYPE == SUPERPOINT)
    {
        ORB_SLAM2::SPextractor extractor(nFeatures, fScaleFactor, nLevels, iniThFAST, minThFAST);
        extractor_ptr = &extractor;
        MAX_DISTANCE_BETWEEN_DESCRIPTORS = 1.4142135623730951;
    }
    else if (FEATURE_EXTRACTOR_TYPE == SIFT)
    {
        ORB_SLAM2::SIFTextractor extractor(nFeatures, fScaleFactor, nLevels, iniThFAST, minThFAST);
        extractor_ptr = &extractor;
        MAX_DISTANCE_BETWEEN_DESCRIPTORS = 1.4142135623730951;
    }
    else
    {
        std::invalid_argument("Unknown feature extractor type. Exiting ...");
    }

    std::cout << "Feature extractor "<< FEATURE_EXTRACTOR_TYPE << " has been set." << std::endl;

    cv::Mat mDescriptors;
    std::vector<cv::KeyPoint> mvKeys;

    int success_semantic = 0;
    int success_normal = 0;

    std::cout << "Testing loop is starting ..." << std::endl;

    for (int i = 0; i < NUMBER_OF_REFERENCES; i++)
    {
        // To track the process
        if (i % 10 == 0)
        {
            std::cout << "<" << std::flush;
        }

        int idx = indices[i];

        cv::Mat im1 = cv::imread(vstrImageFilenames[idx], 0);
        cv::Mat im1_sem = cv::imread(vstrSemFilenames[idx], 0);

        cv::Mat P1_row = GTPoses.row(idx);
        ORB_SLAM2::Frame frame1(im1, im1_sem, extractor_ptr, K, DISTORTION_COEFFICIENTS);

        // ORB-SLAM2 initilization module
        ORB_SLAM2::Initializer *mpInitializer = new ORB_SLAM2::Initializer(frame1, 1.0, 200);

        for (int j = 0; j < INTERVALS.size(); j++)
        {

            std::vector<int> matches_semantic, matches_normal;
            cv::Mat GT_R, GT_t;
            cv::Mat Sem_R, sem_t;
            cv::Mat Normal_R, normal_t;

            // These variables are not used. They are defined just to run the initilization functions.
            std::vector<cv::Point3f> dummy1, dummy2;
            std::vector<bool> dumdum1, dumdum2;

            cv::Mat im2 = cv::imread(vstrImageFilenames[idx + INTERVALS[j]], 0);
            cv::Mat im2_sem = cv::imread(vstrSemFilenames[idx + INTERVALS[j]], 0);
            cv::Mat P2_row = GTPoses.row(idx + INTERVALS[j]);

            ORB_SLAM2::Frame frame2(im2, im2_sem, extractor_ptr, K, DISTORTION_COEFFICIENTS);

            calculateGT(P1_row, P2_row, GT_R, GT_t);

            int match_num_sem = matcher(frame1, frame2, matches_semantic, true);

            if (match_num_sem > 10)
            {

                if (mpInitializer->Initialize(frame2, matches_semantic, Sem_R, sem_t, dummy1, dumdum1))
                {
                    rot_error_sem.push_back(rot_error(Sem_R, GT_R));
                    trans_error_sem.push_back(trans_error(sem_t, GT_t));

                    success_semantic++;
                }
            }

            int match_num_normal = matcher(frame1, frame2, matches_normal, false);

            if (match_num_normal > 10)
            {

                if (mpInitializer->Initialize(frame2, matches_normal, Normal_R, normal_t, dummy2, dumdum2))
                {
                    rot_error_normal.push_back(rot_error(Normal_R, GT_R));
                    trans_error_normal.push_back(trans_error(normal_t, GT_t));
                    success_normal++;
                }
            }
        }
    }

    float rmse_rot_sem = calculateRMS(rot_error_sem);
    float rmse_trans_sem = calculateRMS(trans_error_sem);
    float rmse_rot_normal = calculateRMS(rot_error_normal);
    float rmse_trans_normal = calculateRMS(trans_error_normal);

    std::cout << std::endl;
    std::cout << "rmse_rot_sem: " << rmse_rot_sem << std::endl;
    std::cout << "rmse_trans_sem: " << rmse_trans_sem << std::endl;
    std::cout << "rmse_rot_normal: " << rmse_rot_normal << std::endl;
    std::cout << "rmse_trans_normal: " << rmse_trans_normal << std::endl;

    std::cout << "Sem success: " << success_semantic << std::endl;
    std::cout << "Normal success: " << success_normal << std::endl;

    std::cout << "Ended! The tested directory is " << dataSetDirectoryPath << std::endl;
    std::cout << "            ************************************       " << std::endl;
    std::cout << "     ***************************************************    " << std::endl;
    std::cout << " **************************************************************     " << std::endl;

    return 0;
};

int matcher(ORB_SLAM2::Frame &F1, ORB_SLAM2::Frame &F2, std::vector<int> &vnMatches12, bool withSemantic)
{

    std::vector<cv::Point2f> vbPrevMatched;
    std::vector<cv::Mat> vbPrevMatched_semanticRegions;
    vbPrevMatched.resize(F1.mvKeysUn.size());
    vbPrevMatched_semanticRegions.resize(F1.mvKeysUn.size());
    for (size_t i = 0; i < F1.mvKeysUn.size(); i++)
    {
        vbPrevMatched[i] = F1.mvKeysUn[i].pt;
        vbPrevMatched_semanticRegions[i] = F1.mvKeysSemanticRegions[i];
    }

    int windowSize = 300;
    bool mbCheckOrientation = true;

    int nmatches = 0;
    vnMatches12 = std::vector<int>(F1.mvKeysUn.size(), -1);

    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    std::vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
    std::vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

    for (size_t i1 = 0, iend1 = F1.mvKeysUn.size(); i1 < iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if (level1 > 0)
        {
            continue;
        }
        // Depending on semantic information usage, generate indices of the features inside the search area
        // Search Space Reduction
        std::vector<size_t> vIndices2;

        vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

        if (vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        float bestDist = FLT_MAX;
        float bestDist2 = FLT_MAX;
        int bestIdx2 = -1;

        for (std::vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            float dist = static_cast<float>(DescriptorDistance(d1, d2));

            if (withSemantic)
            {
                // Photometric and Semantic error combination.
                // Formula 6 in paper.
                float dist_semantic = static_cast<float>(compare_semantics(vbPrevMatched_semanticRegions[i1], F2.mvKeysSemanticRegions[i2]));
                dist = (1 - 0.1) * dist + 0.1 * MAX_DISTANCE_BETWEEN_DESCRIPTORS / float(vbPrevMatched_semanticRegions[i1].size[1]) * dist_semantic;
            }

            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist <= FEATURE_DIST_TRESHOLD)
        {
            if (bestDist < (float)bestDist2 * mfNNratio)
            {
                if (vnMatches21[bestIdx2] >= 0)
                {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchedDistance[bestIdx2] = bestDist;
                nmatches++;

                if (mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }
    }

    if (mbCheckOrientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
            {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0)
                {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
};

void ComputeThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
};

float DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{

    if (FEATURE_EXTRACTOR_TYPE == SUPERPOINT)
    {
        float dist = (float)cv::norm(a, b, cv::NORM_L2);
        return dist;
    }
    else if (FEATURE_EXTRACTOR_TYPE == ORB)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }
};

std::vector<std::filesystem::path> getImageFileNames(const std::filesystem::path &directoryPath)
{
    std::vector<std::filesystem::path> imageFileNames;

    // Supported image extensions
    const std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"};

    // Check if the directory exists
    if (!std::filesystem::exists(directoryPath) || !std::filesystem::is_directory(directoryPath))
    {
        std::cerr << "Invalid directory path: " << directoryPath << std::endl;
        return imageFileNames;
    }

    // Iterate over the directory entries
    for (const auto &entry : std::filesystem::directory_iterator(directoryPath))
    {
        if (entry.is_regular_file())
        {
            std::string extension = entry.path().extension().string();
            if (std::find(imageExtensions.begin(), imageExtensions.end(), extension) != imageExtensions.end())
            {
                imageFileNames.push_back(directoryPath / entry.path().filename().string());
            }
        }
    }

    // Sort the file names in alphabetical order
    std::sort(imageFileNames.begin(), imageFileNames.end());

    return imageFileNames;
}

unsigned int compare_semantics(const cv::Mat &m1, const cv::Mat &m2)
{
    unsigned int len = m1.size[1];
    unsigned int dist = 0;
    for (int i = 0; i < len; i++)
    {
        if (m1.at<uchar>(i) != m2.at<uchar>(i))
        {
            dist++;
        }
    }
    return dist;
};

cv::Mat loadTxtFile(const std::filesystem::path &filePath)
{
    std::ifstream file(filePath);
    std::string line;

    std::vector<float> data;
    while (getline(file, line))
    {
        std::stringstream ss(line);
        float value;
        while (ss >> value)
        {
            data.push_back(value);
        }
    }

    int rows = data.size() / 12;
    cv::Mat mat(rows, 12, CV_32F);
    for (int i = 0; i < rows; i++)
    {
        mat.at<float>(i, 0) = data[i * 12];
        mat.at<float>(i, 1) = data[i * 12 + 1];
        mat.at<float>(i, 2) = data[i * 12 + 2];
        mat.at<float>(i, 3) = data[i * 12 + 3];
        mat.at<float>(i, 4) = data[i * 12 + 4];
        mat.at<float>(i, 5) = data[i * 12 + 5];
        mat.at<float>(i, 6) = data[i * 12 + 6];
        mat.at<float>(i, 7) = data[i * 12 + 7];
        mat.at<float>(i, 8) = data[i * 12 + 8];
        mat.at<float>(i, 9) = data[i * 12 + 9];
        mat.at<float>(i, 10) = data[i * 12 + 10];
        mat.at<float>(i, 11) = data[i * 12 + 11];
    }

    return mat;
};

float rot_error(cv::Mat R_est, cv::Mat R_GT)
{
    cv::Mat R = R_est * R_GT.t();
    double theta = (cv::trace(R)[0] - 1.0) / 2.0;
    theta = std::acos(std::max(std::min(theta, 1.0), -1.0));
    return theta * (180.0 / CV_PI);
};

float trans_error(cv::Mat t_est, cv::Mat t_GT)
{

    t_GT = t_GT / cv::norm(t_GT);
    t_est = t_est / cv::norm(t_est);

    double cos_theta = std::min(std::max(double(cv::Mat(t_est.t() * t_GT).at<float>(0)), -1.0), 1.0);

    return acos(cos_theta) * (180.0 / CV_PI);
};

void calculateGT(cv::Mat pos_row1, cv::Mat pos_row2, cv::Mat &R_GT, cv::Mat &t_GT)
{

    cv::Mat P1 = cv::Mat::eye(4, 4, CV_32FC1);
    cv::Mat P2 = cv::Mat::eye(4, 4, CV_32FC1);

    P1.at<float>(0, 0) = pos_row1.at<float>(0);
    P1.at<float>(0, 1) = pos_row1.at<float>(1);
    P1.at<float>(0, 2) = pos_row1.at<float>(2);
    P1.at<float>(0, 3) = pos_row1.at<float>(3);
    P1.at<float>(1, 0) = pos_row1.at<float>(4);
    P1.at<float>(1, 1) = pos_row1.at<float>(5);
    P1.at<float>(1, 2) = pos_row1.at<float>(6);
    P1.at<float>(1, 3) = pos_row1.at<float>(7);
    P1.at<float>(2, 0) = pos_row1.at<float>(8);
    P1.at<float>(2, 1) = pos_row1.at<float>(9);
    P1.at<float>(2, 2) = pos_row1.at<float>(10);
    P1.at<float>(2, 3) = pos_row1.at<float>(11);

    P2.at<float>(0, 0) = pos_row2.at<float>(0);
    P2.at<float>(0, 1) = pos_row2.at<float>(1);
    P2.at<float>(0, 2) = pos_row2.at<float>(2);
    P2.at<float>(0, 3) = pos_row2.at<float>(3);
    P2.at<float>(1, 0) = pos_row2.at<float>(4);
    P2.at<float>(1, 1) = pos_row2.at<float>(5);
    P2.at<float>(1, 2) = pos_row2.at<float>(6);
    P2.at<float>(1, 3) = pos_row2.at<float>(7);
    P2.at<float>(2, 0) = pos_row2.at<float>(8);
    P2.at<float>(2, 1) = pos_row2.at<float>(9);
    P2.at<float>(2, 2) = pos_row2.at<float>(10);
    P2.at<float>(2, 3) = pos_row2.at<float>(11);

    cv::Mat P2_inv = P2.inv(1);

    cv::Mat P21 = P2_inv * P1;

    P21.rowRange(0, 3).colRange(0, 3).copyTo(R_GT);
    P21.rowRange(0, 3).col(3).copyTo(t_GT);
};

float calculateRMS(std::vector<float> &v)
{
    float sum = 0.0;
    for (float f : v)
    {
        sum += f * f;
    }
    float rms = std::sqrt(sum / v.size());
    return rms;
}

std::vector<int> linspace(int start_in, int end_in, int num_in)
{

    std::vector<int> linspaced;

    int start = start_in;
    int end = end_in;
    int num = num_in;

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(int(std::floor(start + delta * i)));
    }
    linspaced.push_back(end); // I want to ensure that start and end
                              // are exactly the same as the input
    return linspaced;
}