/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

#include <numeric>
#include <boost/circular_buffer.hpp>

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    //vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);
    bool bVis = false;            // visualize results


    /* MAIN LOOP OVER ALL IMAGES */

    // Define detector/descriptor type vectors for automated testing:
    vector<string> detectors, descriptors;
    detectors.push_back("SIFT");
    detectors.push_back("HARRIS");
    detectors.push_back("SHITOMASI");
    detectors.push_back("FAST");    // Detector only
    detectors.push_back("BRISK");
    detectors.push_back("ORB");
    detectors.push_back("AKAZE");  
    
    descriptors.push_back("SIFT");
    descriptors.push_back("BRISK");
    descriptors.push_back("BRIEF");
    descriptors.push_back("ORB"); // Does not work with SIFT detector
    descriptors.push_back("FREAK");
    descriptors.push_back("AKAZE"); // Only works with AKAZE keypoints

    std::vector<double> detectorTimes, descriptorTimes;
    std::vector<int> matchedKeypoints;

    for (string detectorType : detectors) {
        for (string descriptorType : descriptors) {

            // Exception for AKAZE detector/descriptor combination:
            if ( !(detectorType.compare("AKAZE") == 0) != !(descriptorType.compare("AKAZE") == 0) ) continue;
            
            // Exception for SIFT / ORB combination:
            if ( (detectorType.compare("SIFT") == 0) && (descriptorType.compare("ORB") == 0) ) continue;


            // Clear all buffer contents. Added for the automated testing loop:
            dataBuffer.clear(); 
            detectorTimes.clear();
            descriptorTimes.clear();
            matchedKeypoints.clear();


            std::cout   << "***************************************" << endl
                        << "Detector: " << detectorType << endl 
                        << "Descriptor: " << descriptorType << endl
                        << "***************************************" << endl;


            // Original loop starts here:
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */
                cout << "Analyzing image " << imgIndex << " of " << imgEndIndex << endl;

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                // Using boost library's circular_buffer since this one is already very efficient and reliable.
                // As a result, we do not need to delete the following line:
                dataBuffer.push_back(frame);


                //// EOF STUDENT ASSIGNMENT
                cout << "\t#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                //string detectorType = "SIFT";

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, detectorTimes, bVis);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, detectorTimes, bVis);
                }
                else if (detectorType.compare("FAST") == 0 || detectorType.compare("BRISK") == 0 || detectorType.compare("ORB") == 0 || 
                            detectorType.compare("AKAZE") == 0 || detectorType.compare("SIFT") == 0)
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, detectorTimes, bVis);
                }
                else
                {
                    cerr << "Unknown detector type " << detectorType << ". Abort." << endl;
                    return 1;
                }
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = false;
                cv::Rect vehicleRect(535, 180, 180, 150);
                
                std::cout << "\tTotal number of detected keypoints: " << keypoints.size() << endl;

                if (bFocusOnVehicle)
                {
                    // Iterate over the vector of all detected keypoints and keep only those within vehicleRect:
                    std::vector<cv::KeyPoint> keypointsTmp;
                    for (auto kp : keypoints)
                    {
                        if (vehicleRect.contains(kp.pt)) {
                            keypointsTmp.push_back(kp);
                        }
                    }
                    keypoints = keypointsTmp;
                }
                
                std::cout << "\tDetected keypoints in ROI: " << keypoints.size() << endl;
                                
                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    std::cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                std::cout << "\t#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                //string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, descriptorTimes);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "\t#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    //string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG -- Note: Naming of this variable collides with the above one. Bad practice!
                    string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    std::cout << "\t#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    matchedKeypoints.push_back(matches.size());
                    std::cout << "\tNumber of matched keypoints: " << matches.size() << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        std::cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }

            } // eof loop over all images


            double avgDetectorTime = 0.;
            double avgDescriptorTime = 0.;
            double avgMatchedKpts = 0;

            avgDetectorTime = std::accumulate(detectorTimes.begin(), detectorTimes.end(), 0.) / detectorTimes.size();
            avgDescriptorTime = std::accumulate(descriptorTimes.begin(), descriptorTimes.end(), 0.) / descriptorTimes.size();
            avgMatchedKpts = std::accumulate(matchedKeypoints.begin(), matchedKeypoints.end(), 0.) / matchedKeypoints.size();


            std::cout   << "***************************************" << endl
                        << "Avg. detector time: " << avgDetectorTime << "ms" << endl 
                        << "Avg. descriptor time: " << avgDescriptorTime << "ms" << endl
                        << "Avg. number of matched keypoints: " << avgMatchedKpts << endl
                        << "***************************************" << endl
                        << "***************************************" << endl << endl << endl;

        }
    }
    
    
    
    return 0;
}
