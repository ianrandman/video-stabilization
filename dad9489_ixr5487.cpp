/**
 * Perform video stabilization using affine transformations.
 * Track points across a video, build up transformations between pairs of frames,
 * sum them to build a trajectory, smooth the trajectory, then apply smoothed transformations.
 */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <vector>
#include <numeric>
#include <iostream>


using namespace cv;
using namespace std;


struct Transformation {
    // represents a transformation matrix between two frames
    // only accounts for translation and angle

    Transformation() {}

    // constructor given values
    Transformation(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    // constructor given matrix
    Transformation(Mat &transformationMatrix) {
        // decompose into translation and rotation
        dx = transformationMatrix.at<double>(0,2);
        dy = transformationMatrix.at<double>(1,2);
        da = atan2(transformationMatrix.at<double>(1,0), transformationMatrix.at<double>(0,0));
    }

    double dx;
    double dy;
    double da; // angle

    // reproduce matrix given values
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
    // smooth a trajectory, which is a cumulative sum of transformations

    vector<Transformation> smoothedTrajectory;

    // iterate over each frame in the trajectory
    for (int idx = 0; idx < trajectory.size(); idx++) {
        // calculate average trajectory for trajectory at idx by smoothing it with surrounding trajectory

        double x = 0;
        double y = 0;
        double a = 0;
        int count = 0;

        // average over a window defined by smoothRadius
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

        // make new smoothed trajectory
        smoothedTrajectory.emplace_back(Transformation(x, y, a));
    }

    return smoothedTrajectory;
}

void evaluate(vector<Mat> &unstabilizedFrames, vector<Mat> &stabilizedFrames) {
    // unused evaluation metric for crop

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    BFMatcher bfMatcher;
    Ptr<SIFT> siftPtr = SIFT::create();
    int minMatchCount = 10;
    double ratio = 0.7;

    vector<double> croppingRatios;
    vector<double> distortionValues;

    Mat unstabilizedFrame, stabilizedFrame;
    vector<KeyPoint> unstabilizedKeyPoints, stabilizedKeyPoints;
    Mat unstabilizedDescriptors, stabilizedDescriptors;
    vector<vector<DMatch>> matches;
    for (int idx = 0; idx < unstabilizedFrames.size(); idx++) {
        unstabilizedFrame = unstabilizedFrames[idx];
        stabilizedFrame = stabilizedFrames[idx];

        siftPtr->detectAndCompute(unstabilizedFrame, noArray(), unstabilizedKeyPoints, unstabilizedDescriptors);
        siftPtr->detectAndCompute(stabilizedFrame, noArray(), stabilizedKeyPoints, stabilizedDescriptors);

        bfMatcher.knnMatch(unstabilizedDescriptors, stabilizedDescriptors, matches, 2);

        vector<DMatch> goodMatches;
        for (vector<DMatch> matchPair : matches) {
            if (matchPair.size() < 2) {
                continue;
            }

            DMatch unstabilizedMatch = matchPair[0];
            DMatch stabilizedMatch = matchPair[1];

            if (unstabilizedMatch.distance < ratio * stabilizedMatch.distance) {
                goodMatches.push_back(unstabilizedMatch);
            }
        }

        if (goodMatches.size() < minMatchCount) {
            continue;
        }

        vector<Point2f> sourcePoint2fs, destinationPoints2fs;
        for (DMatch match: goodMatches) {
            sourcePoint2fs.push_back(unstabilizedKeyPoints[match.queryIdx].pt);
            destinationPoints2fs.push_back(stabilizedKeyPoints[match.trainIdx].pt);
        }

        Mat homography = estimateRigidTransform(sourcePoint2fs, destinationPoints2fs, false);
        if (homography.data == nullptr) {
            continue;
        }

        double scale = sqrt(
                pow(homography.at<double>(0, 0), 2) +
                        pow(homography.at<double>(0, 1), 2));

        croppingRatios.push_back(1 / scale);

        cout << "Evaluated " << idx + 1 << "/" << unstabilizedFrames.size() << endl;
        int x=1;
    }

    double crop = accumulate(croppingRatios.begin(), croppingRatios.end(), 0.0) / croppingRatios.size();
    cout << "CROP: " << crop << endl;
}


int stabilizeVideo(string filePath, string outDir) {
    // open a video, stabilize it, then output the new video side-by-side with the old

    // open the video
    VideoCapture capture(samples::findFile(filePath));
    if (!capture.isOpened()) {
        // error in opening the video input
        cerr << "Unable to open: " << filePath << endl;
        return 1;
    }

    // iterate over all frames in the video
    int frameNum = 0;
    Mat  prevFrameGray;
    Mat currFrame, currFrameGray;
    vector<Point2f> prevCorners, currCorners;
    vector<Transformation> transformations;
    int nFrames = int(capture.get(CAP_PROP_FRAME_COUNT));
    vector<Mat> unstabilizedFrameGrays;
    while (true) {
        // load frame
        capture >> currFrame;
        if (currFrame.empty()) {
            break;  // no more frames
        }
        unstabilizedFrameGrays.push_back(currFrame);

        // convert to grayscale
        cvtColor(currFrame, currFrameGray, COLOR_BGR2GRAY);

        // find tracking features and store them
        int maxCorners = 150;
        double qualityLevel = 0.01;
        double minDistance = 30;
        int blockSize = 3, gradientSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;
        goodFeaturesToTrack(currFrameGray, // TODO maybe use SIFT
            currCorners,
            maxCorners,
            qualityLevel,
            minDistance,
            Mat(),
            blockSize,
            gradientSize,
            useHarrisDetector,
            k);

        // need both current and previous frame
        // use previous transformation if there is an issue
        if (!prevFrameGray.empty()) {
            Mat transformMatrix;
            if (!prevCorners.empty() && !currCorners.empty()) {
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

                if (!prevCorners.empty() && !currCorners.empty()) {

                    // compute the transformation matrix between the previous frame and the current frame
                    // fullAffine=true for more degrees of freedom
                    transformMatrix = estimateRigidTransform(prevCorners, currCorners, true);
                    // transformation matrix may be null; use previous
                    if (transformMatrix.data == nullptr) {
                        transformMatrix = transformations[transformations.size() - 1].getTransformMatrix();
                    }
                } else {
                    transformMatrix = transformations[transformations.size() - 1].getTransformMatrix();
                }

            } else {
                transformMatrix = transformations[transformations.size() - 1].getTransformMatrix();
            }
            transformations.emplace_back(Transformation(transformMatrix));
        }

        // copy current stuff to previous stuff
        currFrameGray.copyTo(prevFrameGray);
        prevCorners = currCorners;  // TODO maybe clone

        frameNum++;
        cout << "Processed frame " << frameNum << "/" << nFrames << endl;

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

    // smooth the trajectory over multiple radii
    vector<Transformation> smoothedTrajectory = smoothTrajectory(trajectory, 64);
    //    for (auto radius: {4, 8, 16, 64}) {
    //        smoothedTrajectory = smoothTrajectory(smoothedTrajectory, radius);
    //    }
    for (auto radius : { 32, 16, 8, 4 }) {
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
    String outFilename = outDir + "/" + filePath.substr(filePath.find_last_of("/\\") + 1);
    outputVideo.open(outFilename, outputVideo.fourcc('M', 'J', 'P', 'G'), capture.get(CAP_PROP_FPS),
        Size(prevFrameGray.cols, prevFrameGray.rows / 2));

    // stabilize each frame
    Mat unstabilizedFrame, stabilizedFrame, stabilizedFrameGray, displayFrame;
    frameNum = 0;
    vector<Mat> stabilizedFrameGrays;
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

        // convert to grayscale
        cvtColor(stabilizedFrame, stabilizedFrameGray, COLOR_BGR2GRAY);
        stabilizedFrameGrays.push_back(stabilizedFrameGray);

        // put the unstabilized and stabilized videos side by side
        hconcat(unstabilizedFrame, stabilizedFrame, displayFrame);
        resize(displayFrame, displayFrame, Size(displayFrame.cols/2, displayFrame.rows/2));
        //imshow("Unstabilized and Stabilized", displayFrame);
        
//        // wait by the frame period of the original video (converted to milliseconds)
//        waitKey(int(1000.0 * (1 / capture.get(CAP_PROP_FPS))));
        // write to the output
        outputVideo << displayFrame;

        frameNum++;
    }

    //evaluate(unstabilizedFrameGrays, stabilizedFrameGrays);

    return 0;
}


int main( int argc, char** argv ) {
    
    const string dataDir = "data";
    const string unstableDir = dataDir + "/DeepStab/unstable";

    string outDir = dataDir + "/DeepStab/out";

    // gather unstable image filenames
    String folder = unstableDir + "/*.avi";
    vector<String> filepaths;
    glob(folder, filepaths);

    // stabilize each of the unstable videos and store in outDir
    int counter = 0;
    for (String filepath: filepaths) {
        counter++;
        cout << "Beginning processing of video " << counter << "/" << filepaths.size() << endl;
        stabilizeVideo(filepath, outDir);
    }

    return 0;
}
