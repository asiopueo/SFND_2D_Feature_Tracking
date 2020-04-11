#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Selects feature detector/descriptor:
cv::Ptr<cv::Feature2D> detectorDescriptorFactory(const string type)
{
    if (type.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        return cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (type.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool use_orientation = false;
        return cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if (type.compare("ORB") == 0)
    {
        int nfeatures = 500;
        float scaleFactor = 1.2;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        int patchSize = 31;
        int fastThreshold = 20;
        return cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
    }
    else if (type.compare("FREAK") == 0)
    {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0;
        int nOctaves = 4;
        return cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);        
    }
    else if (type.compare("AKAZE") == 0)
    {
        int descriptor_size = 0;
        int desciptor_channels = 3;
        float threshold = 0.001;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        // Somehow doesn't work this way, although both should be the same by default:
        //return cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, descriptor_size, descriptor_size, threshold, nOctaves, nOctaveLayers, cv::KAZE::DIFF_PM_G2);
        return cv::AKAZE::create();
    }
    else if (type.compare("SIFT") == 0)
    {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        return cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else if (type.compare("FAST") == 0)
    {
        return cv::FastFeatureDetector::create(10, true);
    }
    else if (type.compare("BRISK") == 0)
    {
        int thresh = 30;
        int octaves = 3;
        float patternScale = 1.0f;
        return cv::BRISK::create(thresh, octaves, patternScale);
    }
    else
    {
        cerr << "Unknown detector/descriptor type " << type << "." << endl;
        return cv::Ptr<cv::Feature2D>();
    }
}


// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    if (matcherType.compare("MAT_BF") == 0)
    {
        if (descriptorType.compare("SIFT") == 0)
        {
            int normType = cv::NORM_L2;
            matcher = cv::BFMatcher::create(normType, crossCheck);
        }
        else // BRIEF, BRISK, ORB, FREAK and AKAZE
        {
            int normType = cv::NORM_HAMMING;
            matcher = cv::BFMatcher::create(normType, crossCheck);    
        }
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::FlannBasedMatcher::create();
    }
    
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    {   
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {   
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);  // k=2
        
        // Lowe's ratio test:
        const float ratio_thresh = 0.8;
        for (size_t i=0; i<knnMatches.size(); ++i)
            if (knnMatches[0][i].distance < ratio_thresh * knnMatches[i][1].distance)
                matches.push_back(knnMatches[i][0]);
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
// BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = detectorDescriptorFactory(descriptorType);
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "\t" << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}





// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    vector<cv::Point2f> corners;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "\tShi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
// For details cf. https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;
    int aperture = 3;
    double k = 0.04;

    double t = (double)cv::getTickCount();
    cv::Mat score, scoreNormalized, scoreRescaled;
    cv::cornerHarris(img, score, blockSize, aperture, k, cv::BORDER_DEFAULT);
    cv::normalize(score, scoreNormalized, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(scoreNormalized, scoreRescaled);
    
    const int minResponse = 100;
    const float maxOverlap = 0;

    for (size_t i=0; i<scoreNormalized.rows; ++i) {
        for (size_t j=0; j<scoreNormalized.cols; ++j) {
            int response = (int) scoreNormalized.at<float>(i, j);
            bool overlap = false;

            if (response > minResponse)
            {
                cv::KeyPoint newKeyPoint(cv::Point2f(j,i), 2 * aperture, -1.0, response, 0, -1);
                // Non-Maximum Suppression
                for(auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    if (cv::KeyPoint::overlap(newKeyPoint, *it) > maxOverlap)
                    {
                        overlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if(!overlap) {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }



    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "\tHarris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using modern detectors
// Options: FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = detectorDescriptorFactory(detectorType);

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "\t" << detectorType << " detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}