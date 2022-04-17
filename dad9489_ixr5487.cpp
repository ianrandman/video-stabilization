/**
 * Perform video stabilization
 */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>

using namespace cv;
using namespace std;


int main( int argc, char** argv ) {
    const string dataDir = "data";
    const string unstable_dir = dataDir + "/DeepStab/unstable";

    const string filePath = unstable_dir + "/" + (string) argv[1];

    VideoCapture capture( samples::findFile( filePath ) );
    if (!capture.isOpened()){
        // error in opening the video input
        cerr << "Unable to open: " << filePath << endl;
        return 0;
    }

    // iterate over all frames in the video
    int frameNum = 0;
    Mat  prevFrameGray;
    Mat currFrame, currFrameGray;
    vector<Point2f> prevCorners, currCorners;
    while (true) {
        // load frame
        capture >> currFrame;
        if (currFrame.empty()) {
            break;
        }

        // convert to grayscale
        cvtColor(currFrame, currFrameGray, COLOR_BGR2GRAY);


        int maxCorners = 150;
        double qualityLevel = 0.01;
        double minDistance = 10;
        int blockSize = 3, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        goodFeaturesToTrack( currFrameGray, // TODO maybe use SIFT
                             currCorners,
                             maxCorners,
                             qualityLevel,
                             minDistance,
                             Mat(),
                             blockSize,
                             gradientSize,
                             useHarrisDetector,
                             k );

        assert(prevCorners != currCorners);


        // need both current and previous frame
        if (!prevFrameGray.empty()) {
            // calculate optical flow
            vector<uchar> status;
            vector<float> err;
            TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
            calcOpticalFlowPyrLK(prevFrameGray, currFrameGray, prevCorners, currCorners, status, err, Size(15, 15), 2, criteria);

            // remove invalid points
            for (int idx = status.size() - 1; idx >= 0; idx--) {
                if (!status[idx]) { // if invalid
                    // remove the points
                    prevCorners.erase(prevCorners.begin() + idx);
                    currCorners.erase(currCorners.begin() + idx);
                }
            }

            Mat transform = estimateRigidTransform(prevCorners, currCorners, true);


        } else {
            // copy current stuff to previous stuff
            currFrameGray.copyTo(prevFrameGray);
            prevCorners = currCorners;  // TODO maybe clone
        }

        frameNum++;
    }


    return 0;
}
