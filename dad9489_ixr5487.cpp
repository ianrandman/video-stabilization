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


struct Transformation {
    Transformation() {}
    Transformation(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    Transformation(Mat &transformationMatrix) {
        // decompose into translation and rotation
        dx = transformationMatrix.at<double>(0,2);
        dy = transformationMatrix.at<double>(1,2);
        da = atan2(transformationMatrix.at<double>(1,0), transformationMatrix.at<double>(0,0));
    }

    double dx;
    double dy;
    double da; // angle

    Mat getTransformMatrix() const {
        Mat transformationMatrix(2,3, CV_64F);

        // Reconstruct transformationMatrix matrix according to new values
        transformationMatrix.at<double>(0, 0) = cos(da);
        transformationMatrix.at<double>(0, 1) = -sin(da);
        transformationMatrix.at<double>(1, 0) = sin(da);
        transformationMatrix.at<double>(1, 1) = cos(da);

        transformationMatrix.at<double>(0, 2) = dx;
        transformationMatrix.at<double>(1, 2) = dy;

        return transformationMatrix;
    }
};

vector<Transformation> smoothTrajectory(vector<Transformation> &trajectory, int smoothRadius) {
    vector<Transformation> smoothedTrajectory;

    for (int idx = 0; idx < trajectory.size(); idx++) {
        // calculate average trajectory for trajectory at idx by smoothing it with surrounding trajectory

        double x, y, a = 0;
        int count = 0;
        for (int jdx = -smoothRadius; jdx <= smoothRadius; jdx++) {
            if ((idx + jdx) >= 0 && (idx + jdx) < trajectory.size()) {
                x += trajectory[idx + jdx].dx;
                y += trajectory[idx + jdx].dy;
                a += trajectory[idx + jdx].da;

                count++;
            }
        }

        x /= count;
        y /= count;
        a /= count;

        smoothedTrajectory.emplace_back(Transformation(x, y, a));
    }

    return smoothedTrajectory;
}


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
    vector<Transformation> transformations;
    int nFrames = int(capture.get(CAP_PROP_FRAME_COUNT));
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
        double minDistance = 30;
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

        // need both current and previous frame
        if (!prevFrameGray.empty()) {
            // calculate optical flow
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(prevFrameGray, currFrameGray, prevCorners, currCorners, status, err);

            // remove invalid points
            for (int idx = status.size() - 1; idx >= 0; idx--) {
                if (!status[idx]) { // if invalid
                    // remove the points
                    prevCorners.erase(prevCorners.begin() + idx);
                    currCorners.erase(currCorners.begin() + idx);
                }
            }

            // compute the transformation matrix between the previous frame and the current frame
            // fullAffine=true for more degrees of freedom
            Mat transformMatrix = estimateRigidTransform(prevCorners, currCorners, true);
            // transformation matrix may be null; use previous
            if (transformMatrix.data == nullptr) {
                transformMatrix = transformations[transformations.size() - 1].getTransformMatrix();
            }

            transformations.emplace_back(Transformation(transformMatrix));
        }

        // copy current stuff to previous stuff
        currFrameGray.copyTo(prevFrameGray);
        prevCorners = currCorners;  // TODO maybe clone

        frameNum++;
        cout << "Processed " << frameNum << "/" << nFrames << endl;
    }

    // trajectory is a cumulative sum of transformations
    vector<Transformation> trajectory;
    trajectory.push_back(transformations[0]);
    for (int idx = 1; idx < transformations.size(); idx++) {
        Transformation transformation = transformations[idx];
        Transformation prevTrajectory = trajectory[idx - 1];

        // add the current transformation to the previous trajectory
        Transformation newTrajectory = Transformation(
                prevTrajectory.dx + transformation.dx,
                prevTrajectory.dy + transformation.dy,
                prevTrajectory.da + transformation.da);
        trajectory.push_back(newTrajectory);
    }

    vector<Transformation> smoothedTrajectory = smoothTrajectory(trajectory, 64);
//    for (auto radius: {4, 8, 16, 64}) {
//        smoothedTrajectory = smoothTrajectory(smoothedTrajectory, radius);
//    }
    for (auto radius: {32, 16, 8, 4}) {
        smoothedTrajectory = smoothTrajectory(smoothedTrajectory, radius);
    }

    vector<Transformation> smoothedTransformations;

    // calculate difference between smoothed trajectory and original trajectory
    for (int idx = 0; idx < trajectory.size(); idx++) {
        double diffX = smoothedTrajectory[idx].dx - trajectory[idx].dx;
        double diffY = smoothedTrajectory[idx].dy - trajectory[idx].dy;
        double diffA = smoothedTrajectory[idx].da - trajectory[idx].da;

        // add this difference back onto original transformations
        smoothedTransformations.emplace_back(Transformation(
                transformations[idx].dx + diffX,
                transformations[idx].dy + diffY,
                transformations[idx].da + diffA
        ));
    }

    // go back to start of video (second frame)
    capture.set(CAP_PROP_POS_FRAMES, 1);

    // initialize output video
    VideoWriter outputVideo;
    outputVideo.open("output.avi", outputVideo.fourcc('M', 'J', 'P', 'G'), 30,
                     Size(prevFrameGray.cols, prevFrameGray.rows / 2));

    Mat unstabilizedFrame, stabilizedFrame, displayFrame;
    frameNum = 0;
    int displayWidth = int(2 * capture.get(CAP_PROP_FRAME_WIDTH));
    int displayHeight = int(capture.get(CAP_PROP_FRAME_HEIGHT));
    while (true) {
        // load unstabilized Frame
        capture >> unstabilizedFrame;
        if (unstabilizedFrame.empty()) {
            break;
        }

        // get the smoothed transformation matrix
        Mat transformationMatrix = smoothedTransformations[frameNum].getTransformMatrix();
        // apply the transformation matrix
        warpAffine(unstabilizedFrame, stabilizedFrame, transformationMatrix, unstabilizedFrame.size());

        hconcat(unstabilizedFrame, stabilizedFrame, displayFrame);
        resize(displayFrame, displayFrame, Size(displayFrame.cols/2, displayFrame.rows/2));
        imshow("Unstabilized and Stabilized", displayFrame);

//        // wait by the frame period of the original video (converted to milliseconds)
//        waitKey(int(1000.0 * (1 / capture.get(CAP_PROP_FPS))));
        waitKey(10);
        outputVideo << displayFrame;

        frameNum++;
    }


    return 0;
}
