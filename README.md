# SFND 2D Feature Tracking
The contents of this repository represent my submission for the midterm project of the camera chapter of Udacity's Sensor Fusion Engineer nanodegree.
The final goal of the camera chapter is to develop a TTC (time-to-collision) estimator based on monocular camera images and Lidar data.

The purpose of this intermediate project is to develop a TTC estimator which only relies on monocular visual information. The detector is based on so-called keypoint detection and matching algorithms.

The following detectors and descriptors have been scrutinized in this project:

**Keypoint detectors:** SIFT, (Harris), (Shi-Tomasi), FAST, BRISK, ORB, and AKAZE, 

**Keypoint descriptors:** SIFT, BRISK, BRIEF, ORB, FREAK, AKAZE

The table below illustrates all the detector/descriptor combination which have tried out in the course of this project and which were functional in a technical sense. Analyzing and understanding their results is one of the goals of this project.

|              | SIFT     | BRISK    | BRIEF   | ORB     | FREAK   |AKAZE|
|:------------:|:--------:|:--------:|:-------:|:-------:|:-------:|:---:|
| SIFT         |  &#8226; |  &#8226; | &#8226; |         |  &#8226;|     |
| (Harris)     |  &#8226; |  &#8226; | &#8226; | &#8226; |  &#8226;|     |
| (Shi-Tomasi) |  &#8226; |  &#8226;  |  &#8226;  |  &#8226; |  &#8226;    |     |
| FAST         |  &#8226; |  &#8226;  |  &#8226;  |  &#8226; |  &#8226;    |     |
| BRISK        |  &#8226; |  &#8226;  |  &#8226;  |  &#8226; |  &#8226;    |     |
| ORB          |  &#8226; |  &#8226;  |  &#8226;  |  &#8226; |  &#8226;    |     |
| AKAZE        |          |             |      |     |      |  &#8226;   |

Notice that the detectors in parentheses (Harris and Shi-Tomasi) were not required for our benchmark testing but where included nontheless. The Harris detector has been implemented as part of this assignment.


# Discussion of Project Tasks

## Data Buffer
### MP.1 Data Buffer Optimization
For the data buffer, our choice fell onto the well-known ring buffer implementation of the *boost library*,

    boost::circular_buffer<T> 
    
because of its readily available and proven implementation. Of course, a simpler solution in form of a simple ´std::vector<T>´ would have satisfied our requirement. Notice that, when using the `vector` class, the size `dataBufferSize` of the buffer can alread be initialized by

    std::vector<DataFrame> dataBuffer(dataBufferSize)

hence circumventing the need for dynamic reallocation of the vector.



## Keypoints
### MP.2 Keypoint Detection
The keypoint detectors have been implemented in the file `matching2D_Student.cpp`. Here, the function 

    cv::Ptr<cv::Feature2D> detectorDescriptorFactory(const string type)

takes a string describing the detector type and returns a pointer for the initialized detector. Internally, a simple if-else-structure selects the detector type, and a final else-clause catches the case when the string has no matching detector or descriptor type.


### MP.3 Keypoint Removal
This assignment requires the removal of keypoints outside a region described by the rectangle

    cv::Rect vehicleRect(535, 180, 180, 150)

This is done by the simple for-loop given by:

    std::vector<cv::KeyPoint> keypointsTmp;
    for (auto kp : keypoints)
    {
        if (vehicleRect.contains(kp.pt)) {
            keypointsTmp.push_back(kp);
        }
    }
    keypoints = keypointsTmp;

Notice that we are using a temporary vector structure instead of utilizing `std::vector::erase()`, hence sacrificing space complexity for time.
Note, that a more sophisticated method to determine the region-of-interest (ROI) using the YOLO detector will be discussed in the following project.

## Descriptors
### MP.4 Keypoint Descriptors
As described in section [MP.2](#mp2-keypoint-detection) above, the descriptors (BRIEF, ORB, FREAK, AKAZE, and SIFT) have been implemented in a unified function `detectorDescriptorFactory()` using the the OpenCV library. Notice that the return is `cv::Ptr<cv::Feature2D>` where `cv::Feature2D` is the common ancestor for all the descriptors.

### MP.5 Descriptor Matching
The descriptor matching procedure is implemented in the function `matchDescriptors()` in `matching2D_Student.cpp`. The function takes, among other arguments, the two strings `matcherType` and `selectorType` which determine the details of the descriptor matching.

FLANN (Fast Library for Approximate Nearest Neighbors) is chosen if the string `matcherType` is set to `MAT_FLANN`. The implementation is covered by the OpenCV library, i.e. by the line
    
    matcher = cv::FlannBasedMatcher::create();

KNN (k-nearest neighbors) is selected when the string `selectorType` is set to `SEL_KNN`. The implementation is given by (for the case of k=2):

    vector<vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descSource, descRef, knnMatches, 2);  // k=2
    

### MP.6 Descriptor Distance Ratio
The descriptor distance ratio test ("Lowe's ratio test") is utilized when the k-NN selector has been chosen ([cf. above](#mp5-descriptor-matching)). It is implented for ratio threshold of `0.8` as follows:

    const float ratio_thresh = 0.8;
    for (size_t i=0; i<knnMatches.size(); ++i)
        if (knnMatches[0][i].distance < ratio_thresh * knnMatches[i][1].distance)
            matches.push_back(knnMatches[i][0]);


## Performance
The results of the performance evaluation have been included in the spreadsheet [Results.ods](./Results.ods) (OpenOffice Calc-file) which is contained in the base directory of this repository.

In order to automate the extensive testing, two outer for-loops have been added which iterate over all eligible detector/descriptor combinations. This simplifies the performance evaluation of all detector/descriptor combinations considerably.

In addition, after iterating over all the images for a particular detector/descriptor combination, the following numbers are calculated:
* Average number of all detected keypoints,
* Average detection time per image,
* Averade descriptor extraction time per image.

### MP.7 Performance Evaluation 1
The number of keypoints on the preceding vehicle for all 10 images have been recorded for all available detectors. The results are available in table `Task MP.7` of [Results.ods](./Results.ods).
Notice that the Harris detector has detected the least amount of keypoints and the FAST detector the most.


### MP.8 Performance Evaluation 2
The number of *all* matched keypoints (not only on the preceding vehicle) for all 10 images have been counted and the average number `avgMatchedKpts` is evaluated at the end of the main loop. Since this process is done automatically, and hence easily reproducible by calling `$>./2D_feature_tracking`, we have entered the average values into the table `Task MP.8` of [Results.ods](./Results.ods).


### MP.9 Performance Evaluation 3
According to our test results, we can rank the top 3 detector/descriptor combinations as follows (less processing time is better):

|Rank| Detector | Descriptor | Processing time [ms] |
|----|----------|------------|:--------------------:|
|1.  | ORB      | BRIEF      | 9.5                  |
|2.  | FAST     | ORB        | 10.4                 |
|3.  | ORB      | BRISK      | 12.6                 |

Of course, such a ranking, performed using the full OpenCV-library on a cloud based VM-host can only give a partial answer. Further testing on embedded hardware and careful evaluation of the results is required until a full answer can be given.


## Further Considerations
The testing of the detectors and descriptors should be further automated and the results directly written to a plain text file (e.g. csv or tsv). In this way, the numbers could be extracted and transferred into the spreadsheets more easily via simple copy & paste.




## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.