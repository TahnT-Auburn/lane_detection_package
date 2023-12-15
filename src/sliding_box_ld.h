//STD
#include <fstream>
#include <sstream>
#include <iostream>
#include <tuple>

//OpenCV
#include <opencv2/opencv.hpp>

//ROS2
#include <rclcpp/rclcpp.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class SlidingBoxLaneDetection : public rclcpp::Node
{
public:
    SlidingBoxLaneDetection()
    : Node("sliding_box_lane_detection")
    {
        parseParameters();
        run();
    }

private:

    //Initialize ROS2 parameters
    float confThreshold;    //YOLO
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    string device_;
    string input_type_;
    string input_path_;
    bool write_output_;
    string output_file_;
    string classesFile;
    String modelConfiguration;
    String modelWeights;
    std::vector<double> camera_matrix_vector_;  //Cam intrinsics
    std::vector<double> dist_coeffs_;

    //Class methods
    void parseParameters();
    void run();
    cv::Mat postprocess(const Mat& frame, const Mat& binary_frame, const vector<Mat>& outs);
    void drawPred(int idx, int classId, float conf, int left, int top, int right, int bottom, const Mat& frame);
    vector<string> getOutputsNames(const Net& net);
    cv::Mat undistortImage(const cv::Mat& frame, const cv::Size& resolution);
    cv::Mat thresholdBinaryImage(const cv::Mat& undist_frame);
    cv::Mat maskImage(const cv::Mat& binary_frame, vector<int> bb_left, vector<int> bb_right, vector<int> bb_top, vector<int> bb_bottom);
    cv::Mat inversePerspectiveMapping(const cv::Mat& frame);
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SlidingBoxLaneDetection>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}