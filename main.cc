#include<iostream>
#include<algorithm>
#include<fstream>
// #include"ORBextractor.h"
// #include"SIFTextractor.h"
#include "SPextractor.h"
#include "Frame.h"
#include "Initializer.h"

#include <cmath>
#include <vector>

#define CV_PI   3.1415926535897932384626433832795

cv::Mat K;
std::string PATH;
int NUMBER_OF_IMAGES;


int NUMBER_OF_REFERENCES = 250;
std::vector<int> INTERVALS{ 5, 10, 15};
cv::Mat DISTCONF = cv::Mat::zeros(4, 1, CV_32F);
std::string directory = "/cluster/scratch/oilter/KITTI/sequences/";
const float TH_HIGH = 0.70;
const float TH_LOW = 0.30;
const int HISTO_LENGTH = 30;
const float mfNNratio = 0.7;


int matcher(ORB_SLAM2::Frame &F1, ORB_SLAM2::Frame &F2, std::vector<int> &vnMatches12, bool withSemantic);
void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
void LoadImages(std::vector<std::string> &vstrImageFilenames, std::vector<std::string> &vstrSemFilenames );
unsigned int compare_semantics(const cv::Mat &m1, const cv::Mat &m2 );
cv::Mat loadTxtFile(string filePath);
void setParams(std::string path);
float rot_error(cv::Mat R_est, cv::Mat R_GT);
float trans_error(cv::Mat t_est, cv::Mat t_GT);
void calculateGT(cv::Mat pos_row1, cv::Mat pos_row2, cv::Mat &R_GT, cv::Mat &t_GT);
float calculateRMS(std::vector<float>& v);
std::vector<int> linspace(int start_in, int end_in, int num_in);


const float max_dist =  1.4142135623730951;


int main(int argc, char **argv){


    PATH = argv[1];

    std::cout<<"Started! The Path ID: " << PATH << std::endl;
    std::cout<<" **************************************************************" <<std::endl;
    std::cout<<"     ***************************************************" <<std::endl;
    std::cout<<"            ***************************************" <<std::endl;


    int nFeatures       = 500;
    float fScaleFactor  = 1.2;
    int nLevels         = 1;
    int fIniThFAST      = 20;
    int fMinThFAST      = 7;
    float iniThFAST     = 0.015;
    float minThFAST     = 0.007;


    std::vector<float> rot_error_normal;
    std::vector<float> trans_error_normal;
    std::vector<float> rot_error_sem;
    std::vector<float> trans_error_sem;


    setParams(PATH);


    std::vector<std::string> vstrImageFilenames;
    std::vector<std::string> vstrSemFilenames;
    cv::Mat GTPoses = loadTxtFile(directory + PATH +".txt");
    LoadImages(vstrImageFilenames, vstrSemFilenames);

    // std::cout<<"GTPoses.rows(): "<< GTPoses.rows <<std::endl;
    // std::cout<<"GTPoses.col(): "<< GTPoses.cols <<std::endl;
    // std::cout<<"vstrImageFilenames: "<< vstrImageFilenames.size() <<std::endl;
    // std::cout<<"vstrSemFilenames: "<< vstrSemFilenames.size() <<std::endl;


    //int interval = std::floor( (NUMBER_OF_IMAGES - INTERVALS[2]) / NUMBER_OF_PAIRS);


    std::vector<int> indices = linspace(0, NUMBER_OF_IMAGES - INTERVALS.back() - 2, NUMBER_OF_REFERENCES);

    //std::cout<<"indices: "<< indices <<std::endl;

    //std::cout<<"indices.back(): "<< indices.back() <<std::endl;

    ORB_SLAM2::SPextractor extractor(nFeatures,fScaleFactor,nLevels,iniThFAST,minThFAST);
    
    ORB_SLAM2::SPextractor *extractor_ptr = &extractor;
    cv::Mat mDescriptors;
    std::vector<cv::KeyPoint> mvKeys;

    int success_semantic = 0;
    int success_normal = 0;


    for (int i=0; i < NUMBER_OF_REFERENCES; i++){

        //std::cout<<"i: "<< i <<std::endl;
        //std::cout<<"indices[i]: "<< indices[i] <<std::endl;

        if( i%10 == 0 ){

            std::cout<<"<"<<std::flush;
        }

        int idx = indices[i];
        
        cv::Mat im1 = cv::imread(vstrImageFilenames[idx], 0);
        cv::Mat im1_sem = cv::imread(vstrSemFilenames[idx], 0);

        //std::cout<<"After read image1" <<std::endl;

        cv::Mat P1_row = GTPoses.row(idx);
        ORB_SLAM2::Frame frame1(im1, im1_sem, extractor_ptr, K, DISTCONF );

        //std::cout<<"After frame1" <<std::endl;

        ORB_SLAM2::Initializer* mpInitializer =  new ORB_SLAM2::Initializer(frame1,1.0,200);

        //std::cout<<"After initializer" <<std::endl;


        for(int j=0; j<INTERVALS.size(); j++){

            std::vector<int> matches_semantic, matches_normal;
            cv::Mat GT_R, GT_t;
            cv::Mat Sem_R, sem_t;
            cv::Mat Normal_R, normal_t;
            std::vector<cv::Point3f> dummy1, dummy2; 
            std::vector<bool> dumdum1, dumdum2;

            //std::cout<<"j: "<< j <<std::endl;
            //std::cout<<"INTERVALS[j]: "<< INTERVALS[j] <<std::endl;

            cv::Mat im2 = cv::imread(vstrImageFilenames[idx + INTERVALS[j] ], 0);
            cv::Mat im2_sem = cv::imread(vstrSemFilenames[idx + INTERVALS[j] ], 0);
            //std::cout<<"After read image2" <<std::endl;
            cv::Mat P2_row = GTPoses.row(idx + INTERVALS[j]);
            //std::cout<<"After GTPoses" <<std::endl;


            ORB_SLAM2::Frame frame2(im2, im2_sem, extractor_ptr, K, DISTCONF );
            //std::cout<<"After frame2" <<std::endl;


            calculateGT(P1_row, P2_row, GT_R, GT_t);
            //std::cout<<"After calculateGT" <<std::endl;

            

            int match_num_sem = matcher(frame1, frame2, matches_semantic, true);
            //std::cout<<"After matcher" <<std::endl;

            ////std::cout<<"match_num_sem: " << match_num_sem << std::endl;

            if(match_num_sem > 10){
            
                if(mpInitializer->Initialize(frame2, matches_semantic, Sem_R, sem_t, dummy1, dumdum1)){
                    
                    //std::cout<< " 11111111111111111111111111 " <<std::endl;

                    rot_error_sem.push_back( rot_error(Sem_R, GT_R)  );
                    trans_error_sem.push_back( trans_error(sem_t, GT_t)  );

                    //std::cout<<"Trans error: "<< trans_error(sem_t, GT_t) <<std::endl;
                    //std::cout<< " 222222222222222222222222222222222 " <<std::endl;

                    success_semantic ++;

                }

                //std::cout<<"After Initialize sem" <<std::endl;


            }

            int match_num_normal = matcher(frame1, frame2, matches_normal, false);

            if(match_num_normal > 10){
                                
                if(mpInitializer->Initialize(frame2, matches_normal, Normal_R, normal_t, dummy2, dumdum2)){

                    rot_error_normal.push_back( rot_error(Normal_R, GT_R)  );
                    trans_error_normal.push_back( trans_error(normal_t, GT_t)  );
                    success_normal ++;
                    
                }

                //std::cout<<"After Initialize normal" <<std::endl;
            }

        }


    }

    //std::cout<<"Before calculateRMS" <<std::endl;

    float rmse_rot_sem      = calculateRMS(rot_error_sem);
    float rmse_trans_sem    = calculateRMS(trans_error_sem);
    float rmse_rot_normal   = calculateRMS(rot_error_normal);
    float rmse_trans_normal = calculateRMS(trans_error_normal);

    //std::cout<<"After calculateRMS" <<std::endl;

    std::cout<<std::endl;
    std::cout<<"rmse_rot_sem: " << rmse_rot_sem << std::endl;
    std::cout<<"rmse_trans_sem: " << rmse_trans_sem << std::endl;
    std::cout<<"rmse_rot_normal: " << rmse_rot_normal << std::endl;
    std::cout<<"rmse_trans_normal: " << rmse_trans_normal << std::endl;

    std::cout<<"Sem success: " << success_semantic << std::endl;
    std::cout<<"Normal success: " << success_normal << std::endl;



    std::cout<<"Ended! The Path ID: " << PATH << std::endl;
    std::cout<<"            ************************************       " <<std::endl;
    std::cout<<"     ***************************************************    " <<std::endl;
    std::cout<<" **************************************************************     " <<std::endl;
    
    
    return 0;



};

int matcher(ORB_SLAM2::Frame &F1, ORB_SLAM2::Frame &F2, std::vector<int> &vnMatches12, bool withSemantic)
{

    std::vector<cv::Point2f> vbPrevMatched;
    std::vector<cv::Mat> vbPrevMatched_semanticRegions;
    vbPrevMatched.resize(F1.mvKeysUn.size());
    vbPrevMatched_semanticRegions.resize(F1.mvKeysUn.size());
    for(size_t i=0; i<F1.mvKeysUn.size(); i++){
        vbPrevMatched[i]=F1.mvKeysUn[i].pt;
        vbPrevMatched_semanticRegions[i]=F1.mvKeysSemanticRegions[i];
    }

    int windowSize = 300;
    bool mbCheckOrientation = true;

    int nmatches=0;
    vnMatches12 = std::vector<int>(F1.mvKeysUn.size(),-1);

    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    std::vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    std::vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0){
            continue;
        }
        // Depending on semantic information usage, generate indices of the features inside the search area
        // Search Space Reduction
        std::vector<size_t> vIndices2;

        vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);                                    

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        float bestDist = FLT_MAX;
        float bestDist2 = FLT_MAX;
        int bestIdx2 = -1;

        for(std::vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            float dist = static_cast<float>(DescriptorDistance(d1,d2));

            if(withSemantic){

                float dist_semantic = static_cast<float>(compare_semantics(vbPrevMatched_semanticRegions[i1], F2.mvKeysSemanticRegions[i2]));
                dist = (1-0.1) * dist + 0.1* max_dist / float(vbPrevMatched_semanticRegions[i1].size[1]) * dist_semantic;
            }

            

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    return nmatches;
};

void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
};

float DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    float dist = (float)cv::norm(a, b, cv::NORM_L2);
    return dist;
};

void LoadImages(std::vector<std::string> &vstrImageFilenames, std::vector<std::string> &vstrSemFilenames )
{
    std::string strPrefixImg = directory + PATH + "/image_0/";
    std::string strPrefixSem = directory + PATH + "/semantic_image_0/";

    vstrImageFilenames.resize(NUMBER_OF_IMAGES);
    vstrSemFilenames.resize(NUMBER_OF_IMAGES);

    for(int i=0; i<NUMBER_OF_IMAGES; i++)
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        vstrImageFilenames[i] = strPrefixImg + ss.str() + ".png";
        vstrSemFilenames[i] = strPrefixSem + ss.str() + ".png";
    }
};

unsigned int compare_semantics(const cv::Mat &m1, const cv::Mat &m2 )
{
    unsigned int len = m1.size[1];
    unsigned int dist = 0;
    for(int i = 0; i < len ; i++){
        if(m1.at<uchar>(i) != m2.at<uchar>(i)){
            dist ++;
        }
    }
    return dist;
};

cv::Mat loadTxtFile(string filePath) {
    std::ifstream file(filePath.c_str());
    std::string line;

    std::vector<float> data;
    while (getline(file, line)) {
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            data.push_back(value);
        }
    }

    int rows = data.size() / 12;
    cv::Mat mat(rows, 12, CV_32F);
    for (int i = 0; i < rows; i++) {
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

void setParams(std::string path){

    if (path == "00" || path == "01" || path == "02"  ){

        cv::Mat dummy = cv::Mat::eye(3,3,CV_32F);
        dummy.at<float>(0,0) = 718.856;
        dummy.at<float>(1,1) = 718.856;
        dummy.at<float>(0,2) = 607.1928;
        dummy.at<float>(1,2) = 185.2157;
        dummy.copyTo(K);

        if(path == "00"){
            NUMBER_OF_IMAGES = 4541;
        }
        else if(path == "02"){
            NUMBER_OF_IMAGES = 4661;
        }

    }
    else if (path == "03" ){

        cv::Mat dummy = cv::Mat::eye(3,3,CV_32F);
        dummy.at<float>(0,0) = 721.5377;
        dummy.at<float>(1,1) = 721.5377;
        dummy.at<float>(0,2) = 609.5593;
        dummy.at<float>(1,2) = 172.854;
        dummy.copyTo(K);

        NUMBER_OF_IMAGES = 801;

    }
    else if (path == "04" || path == "05" || path == "06" || path == "07" || path == "08" || path == "09" || path == "10" ){
        
        cv::Mat dummy = cv::Mat::eye(3,3,CV_32F);
        dummy.at<float>(0,0) = 707.0912;
        dummy.at<float>(1,1) = 707.0912;
        dummy.at<float>(0,2) = 601.8873;
        dummy.at<float>(1,2) = 183.1104;
        dummy.copyTo(K);

        if(path == "04"){
            NUMBER_OF_IMAGES = 271; 
        }
        else if(path == "05"){
            NUMBER_OF_IMAGES = 2761; 
        }
        else if(path == "06"){
            NUMBER_OF_IMAGES = 1101; 
        }
        else if(path == "07"){
            NUMBER_OF_IMAGES = 1101; 
        }
        else if(path == "08"){
            NUMBER_OF_IMAGES = 4071; 
        }
        else if(path == "09"){
            NUMBER_OF_IMAGES = 1591; 
        }
        else if(path == "10"){
            NUMBER_OF_IMAGES = 1201; 
        }

    }

};

float rot_error(cv::Mat R_est, cv::Mat R_GT){
    cv::Mat R = R_est * R_GT.t();
    double theta = (cv::trace(R)[0] - 1.0) / 2.0;
    theta = std::acos(std::max(std::min(theta, 1.0), -1.0));
    return theta * (180.0 / CV_PI);
};

float trans_error(cv::Mat t_est, cv::Mat t_GT){

    t_GT=t_GT/cv::norm(t_GT);
    t_est = t_est / cv::norm(t_est);



    double cos_theta = std::min( std::max( double(cv::Mat(t_est.t() * t_GT).at<float>(0)),  -1.0), 1.0)    ;

    return acos(cos_theta) * (180.0 / CV_PI);
};

void calculateGT(cv::Mat pos_row1, cv::Mat pos_row2, cv::Mat &R_GT, cv::Mat &t_GT){

    cv::Mat P1 = cv::Mat::eye(4,4,CV_32FC1);
    cv::Mat P2 = cv::Mat::eye(4,4,CV_32FC1);


    P1.at<float>(0,0) = pos_row1.at<float>(0);
    P1.at<float>(0,1) = pos_row1.at<float>(1);
    P1.at<float>(0,2) = pos_row1.at<float>(2);
    P1.at<float>(0,3) = pos_row1.at<float>(3);
    P1.at<float>(1,0) = pos_row1.at<float>(4);
    P1.at<float>(1,1) = pos_row1.at<float>(5);
    P1.at<float>(1,2) = pos_row1.at<float>(6);
    P1.at<float>(1,3) = pos_row1.at<float>(7);
    P1.at<float>(2,0) = pos_row1.at<float>(8);
    P1.at<float>(2,1) = pos_row1.at<float>(9);
    P1.at<float>(2,2) = pos_row1.at<float>(10);
    P1.at<float>(2,3) = pos_row1.at<float>(11);
    

    P2.at<float>(0,0) = pos_row2.at<float>(0);
    P2.at<float>(0,1) = pos_row2.at<float>(1);
    P2.at<float>(0,2) = pos_row2.at<float>(2);
    P2.at<float>(0,3) = pos_row2.at<float>(3);
    P2.at<float>(1,0) = pos_row2.at<float>(4);
    P2.at<float>(1,1) = pos_row2.at<float>(5);
    P2.at<float>(1,2) = pos_row2.at<float>(6);
    P2.at<float>(1,3) = pos_row2.at<float>(7);
    P2.at<float>(2,0) = pos_row2.at<float>(8);
    P2.at<float>(2,1) = pos_row2.at<float>(9);
    P2.at<float>(2,2) = pos_row2.at<float>(10);
    P2.at<float>(2,3) = pos_row2.at<float>(11);



    cv::Mat P2_inv = P2.inv(1);

    cv::Mat P21 = P2_inv*P1;

    P21.rowRange(0,3).colRange(0,3).copyTo(R_GT);
    P21.rowRange(0,3).col(3).copyTo(t_GT);

};


float calculateRMS(std::vector<float>& v) {
    float sum = 0.0;
    for (float f : v) {
        sum += f * f;
    }
    float rms = std::sqrt(sum / v.size());
    return rms;
}


std::vector<int> linspace(int start_in, int end_in, int num_in)
{

  std::vector<int> linspaced;

  int start = start_in;
  int end   = end_in;
  int num   = num_in;

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back( int( std::floor(start + delta * i) ) );
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}