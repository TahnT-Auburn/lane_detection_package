//STD
#include <fstream>
#include <sstream>
#include <iostream>
#include <tuple>
#include <numeric>
#include <complex>

//OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

//ROS2
#include "sliding_box_ld.h"

using namespace cv;
using namespace dnn;
using namespace std;

void SlidingBoxLaneDetection::parseParameters()
{
    //Yolo parameters
    this->declare_parameter("conf_threshold");
    this->declare_parameter("nms_threshold");
    this->declare_parameter("inp_width");
    this->declare_parameter("inp_height");
    this->declare_parameter("device");
    this->declare_parameter("input_type");
    this->declare_parameter("input_path");
    this->declare_parameter("write_output");
    this->declare_parameter("output_file");
    this->declare_parameter("classes_file");
    this->declare_parameter("model_configuration");
    this->declare_parameter("model_weights");
    //Camera intrinsics
    this->declare_parameter("camera_matrix_vector");
    this->declare_parameter("dist_coeffs");

    //Get Yolo params
    this->get_parameter("conf_threshold", confThreshold);   
    this->get_parameter("nms_threshold", nmsThreshold);
    this->get_parameter("inp_width", inpWidth);
    this->get_parameter("inp_height", inpHeight);
    this->get_parameter("device", device_);
    this->get_parameter("input_type", input_type_);
    this->get_parameter("input_path", input_path_);
    this->get_parameter("write_output", write_output_);
    this->get_parameter("output_file", output_file_);
    this->get_parameter("classes_file", classesFile);
    this->get_parameter("model_configuration", modelConfiguration);
    this->get_parameter("model_weights", modelWeights);
    //Get camera intrinsics
    this->get_parameter("camera_matrix_vector", camera_matrix_vector_);
    this->get_parameter("dist_coeffs", dist_coeffs_);   

    //Check
    RCLCPP_INFO(this->get_logger(), "Confidence Threshold: %f", confThreshold);
    RCLCPP_INFO(this->get_logger(), "Non-Maximum Suppression Threshold: %f", nmsThreshold);
    RCLCPP_INFO(this->get_logger(), "Input Width: %d", inpWidth);
    RCLCPP_INFO(this->get_logger(), "Input height: %d", inpHeight);
    RCLCPP_INFO(this->get_logger(), "Device: %s", device_.c_str());
}

void SlidingBoxLaneDetection::run()
{   
    //Load Darknet
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    if (device_ == "cpu")
    {
        RCLCPP_INFO(this->get_logger(), "Using %s device", device_.c_str());
        net.setPreferableBackend(DNN_TARGET_CPU);
    }
    else if (device_ == "gpu")
    {   
        RCLCPP_INFO(this->get_logger(), "Using %s device", device_.c_str());
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }

    //Initialize
    cv::VideoCapture cap;
    cv::VideoWriter video;
    cv::Mat frame, blob;
    double fps;

    //Read file
    try
    {
        if (input_type_ == "image")
        {
            //Open image file
            std::ifstream ifile(input_path_);
            if (!ifile)
            {
                RCLCPP_ERROR(this->get_logger(), "No image found");
            }
            cap.open(input_path_);
            RCLCPP_INFO(this->get_logger(), "Image opened successfully\n");
        }
        else if (input_type_ == "video")
        {
            //Open video file
            std::ifstream ifile(input_path_);
            if (!ifile)
            {
                RCLCPP_ERROR(this->get_logger(), "No video found");
            }
            cap.open(input_path_);
            RCLCPP_INFO(this->get_logger(), "Video opened successfully\n");

            //Get video FPS
            fps = cap.get(cv::CAP_PROP_FPS);
        }
    }
    catch (...)
    {
        RCLCPP_ERROR(this->get_logger(), "Cound not open input file");
        rclcpp::shutdown();
    }

    //Resolution
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Size resolution(frame_width, frame_height);

    //Initialize video writer to save output video
    if (input_type_ == "video" && write_output_)
    {
        video.open(output_file_, cv::VideoWriter::fourcc('M','J','P','G'), fps, resolution);
    }
    
    //Process frames
    while (true)
    {
        //Get frame
        cap >> frame;

        //Break if processing is finished
        if (frame.empty())
        {
            RCLCPP_INFO(this->get_logger(), "Processing completed successfully");
            if (write_output_)
            {
                RCLCPP_INFO(this->get_logger(), "Output written to %s", output_file_.c_str());
            }
            cv::waitKey(3000);
            break;
        }

        //Display raw frame
        cv::imshow("Input Frame", frame);
        cv::waitKey(0);

        //----Undistort image----
        cv::Mat undist_frame = undistortImage(frame, resolution);
        //Display undistorted image
        //cv::imshow("Undistorted Frame", undist_frame);
        //cv::waitKey(0);

        //----Generate thresholding binary image----
        cv::Mat binary_frame = thresholdBinaryImage(undist_frame);
        cv::Mat bin_frame_white = binary_frame * 255;
        //Display binary image
        cv::imshow("Binary Frame", bin_frame_white);
        cv::waitKey(0);

        //----Run YOLO and Mask image----
        // Create a 4D blob from a frames
        blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        net.setInput(blob);
        
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        //cv::Mat mask_frame = maskImage(binary_frame);
        cv::Mat mask_frame = postprocess(undist_frame, binary_frame, outs);
        cv::Mat mask_frame_white = mask_frame * 255;
        //Display
        cv::imshow("Masked Frame", mask_frame_white);
        cv::waitKey(0);

        //----Inverse perspective mapping----
        cv::Mat ipm_raw_frame = inversePerspectiveMapping(undist_frame);
        cv::Mat ipm_bin_frame = inversePerspectiveMapping(mask_frame);
        cv::Mat ipm_bin_frame_white = ipm_bin_frame * 255;
        //Display
        cv::imshow("IPM Raw Frame", ipm_raw_frame);
        cv::waitKey(0);
        cv::imshow("IPM Binary Frame", ipm_bin_frame_white);
        cv::waitKey(0);


    }
    //Clean
    cap.release();
    if (input_type_ == "video") video.release();
}

cv::Mat SlidingBoxLaneDetection::postprocess(const Mat& frame, const Mat& binary_frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    //Load names of classes
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    vector<int> bb_left, bb_right, bb_top, bb_bottom;   //Initialize bounding box points
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        string obj_class = classes[classIds[idx]];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int right = box.x + box.width;
        int bottom = box.y + box.height;

        if (obj_class == "car" || obj_class == "truck" || obj_class == "bus")
        {   
            //Draw bounding boxes onto frame
            drawPred(idx, classIds[idx], confidences[idx], left, top,
            right, bottom, frame);

            //Append vehicle bounding box points - used for generating a mask
            bb_left.push_back(left);
            bb_right.push_back(right);
            bb_top.push_back(top);
            bb_bottom.push_back(bottom);
        }
    }

    //Generate mask
    cv::Mat mask_frame = maskImage(binary_frame, bb_left, bb_right, bb_top, bb_bottom);

    return mask_frame;
}

void SlidingBoxLaneDetection::drawPred(int idx, int classId, float conf, int left, int top, int right, int bottom, const Mat& frame)
{   
    //Load names of classes
    vector<string> classes;    
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 65), 1); //Scalar(255, 178, 50)
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = "Tag:" + to_string(idx) + " " + classes[classId] + ":" + label; //
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1*labelSize.height)), Point(left + round(1*labelSize.width), top + baseLine), Scalar(0, 255, 65), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0),1.5);
}

vector<String> SlidingBoxLaneDetection::getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

cv::Mat SlidingBoxLaneDetection::undistortImage(const cv::Mat& frame, const cv::Size& resolution)
{   
    //Camera matrix
    cv::Matx33d camera_matrix_(camera_matrix_vector_[0], camera_matrix_vector_[1], camera_matrix_vector_[2], 
                        camera_matrix_vector_[3], camera_matrix_vector_[4], camera_matrix_vector_[5], 
                        camera_matrix_vector_[6], camera_matrix_vector_[7], camera_matrix_vector_[8]);

    //Precompute lens correction interpolation
    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Matx33f::eye(), camera_matrix_, resolution, CV_32FC1, mapX, mapY);

    //Undistort image
    cv::Mat undist_frame;
    cv::remap(frame, undist_frame, mapX, mapY, cv::INTER_LINEAR);

    return undist_frame;
}

cv::Mat SlidingBoxLaneDetection::thresholdBinaryImage(const cv::Mat& undist_frame)
{
    //Initialize thresholding limit values
    std::vector<int> gradient_thresh, s_thresh, l_thresh, b_thresh, l2_thresh;
    gradient_thresh = {20, 100};
    s_thresh = {80, 255};
    l_thresh = {80, 255};
    b_thresh = {150, 200};
    l2_thresh = {225, 255};

    //Initialize thresholding channels
    cv::Mat hls_frame, lab_frame, luv_frame,
        h, l, s, l2, a, b, l3, u, v; 
    cv::Mat hls_channels[3], lab_channels[3], luv_channels[3];

    //HLS
    cv::cvtColor(undist_frame, hls_frame, cv::COLOR_BGR2HLS);
    cv::split(hls_frame, hls_channels);
    h = hls_channels[0];
    l = hls_channels[1];
    s = hls_channels[2];

    //LAB
    cv::cvtColor(undist_frame, lab_frame, cv::COLOR_BGR2Lab);
    cv::split(lab_frame, lab_channels);
    l2 = lab_channels[0];
    a = lab_channels[1];
    b = lab_channels[2];

    //LUV
    cv::cvtColor(undist_frame, luv_frame, cv::COLOR_BGR2Luv);
    cv::split(luv_frame, luv_channels);
    l3 = luv_channels[0];
    u = luv_channels[1];
    v = luv_channels[2];
    
    //Initialize binary matrices
    cv::Mat binary_b, binary_l;
    //cv::Mat1f binary_bl = cv::Mat1f::zeros(undist_frame.rows, undist_frame.cols);
    cv::Mat binary_bl = cv::Mat::zeros(undist_frame.rows, undist_frame.cols, CV_8UC1);

    //Initialize pixel value pointers
    std::vector<int> pixel_values_b, pixel_values_l3;
    uchar* ptr_b = b.data;
    uchar* ptr_l3 = l3.data;
    uchar* ptr_bl = binary_bl.data;

    //Generate binary image
    for (int i = 0; i < undist_frame.rows; ++i)
    {   
        ptr_b = b.ptr<uchar>(i);
        ptr_l3 = l3.ptr<uchar>(i);
        ptr_bl = binary_bl.ptr<uchar>(i);

        for (int j = 0; j < undist_frame.cols; ++j)
        {   
            pixel_values_b.push_back(ptr_b[j]);
            pixel_values_l3.push_back(ptr_l3[j]);
            
            //B/L channel conditions
            if (((ptr_b[j] > b_thresh[0]) && (ptr_b[j] <= b_thresh[1])) ||
                ((ptr_l3[j] > l2_thresh[0]) && (ptr_l3[j] <= l2_thresh[1])))
            {   
                ptr_bl[j] = 1;
                //binary_bl[j] = 1;
            }
            else
            {   
                ptr_bl[j] = 0;
                //binary_bl[j] = 0;
            }
        }
    }

    return binary_bl;
}

cv::Mat SlidingBoxLaneDetection::maskImage(const cv::Mat& binary_frame, vector<int> bb_left, vector<int> bb_right, vector<int> bb_top, vector<int> bb_bottom)
{   
    //Generate mask image
    cv::Mat mask(binary_frame.size().height, binary_frame.size().width, CV_8UC1, cv::Scalar(0));
    //mask = 0.0; //Initilized as black image

    //Define boundary points
    int tolerance = 275; //tunable
    float height_scale = 4.5; //tunable
    cv::Point p1 = cv::Point(0,binary_frame.size().height);                                             //Bottom-left point
    cv::Point p2 = cv::Point(binary_frame.size().width/2 - tolerance, binary_frame.size().height/height_scale);  //Top-left point
    cv::Point p3 = cv::Point(binary_frame.size().width/2 + tolerance, binary_frame.size().height/height_scale);  //Top-right point
    cv::Point p4 = cv::Point(binary_frame.size().width, binary_frame.size().height);                    //Bottom-right point 

    //Fill mask
    cv::Point vertice_points[] = {p1, p2, p3, p4};
    std::vector<cv::Point> vertices(vertice_points, vertice_points + sizeof(vertice_points) / sizeof(cv::Point));
    std::vector<std::vector<cv::Point>> vertices_to_fill;
    vertices_to_fill.push_back(vertices);
    cv::fillPoly(mask, vertices_to_fill, cv::Scalar(255));

    //Eliminate vehicle detections from mask
    int detect_count = bb_left.size();
    for (int i = 0; i < detect_count; ++i)
    {
        //Fill mask
        p1 = cv::Point(bb_left[i], bb_bottom[i]);
        p2 = cv::Point(bb_left[i], bb_top[i]);
        p3 = cv::Point(bb_right[i], bb_top[i]);
        p4 = cv::Point(bb_right[i], bb_bottom[i]);
        cv::Point vertice_points[] = {p1, p2, p3, p4};
        std::vector<cv::Point> vertices(vertice_points, vertice_points + sizeof(vertice_points) / sizeof(cv::Point));
        vertices_to_fill = {};
        vertices_to_fill.push_back(vertices);
        cv::fillPoly(mask, vertices_to_fill, cv::Scalar(0));
    }


    //Display mask
    cv::imshow("Mask", mask);
    cv::waitKey(0);

    //Apply mask image
    cv::Mat mask_frame = binary_frame.clone();
    cv::bitwise_and(binary_frame, mask, mask_frame);

    return mask_frame;
}

cv::Mat SlidingBoxLaneDetection::inversePerspectiveMapping(const cv::Mat& frame)
{   

    //Define source points (set same as mask points - tunable)
    int tolerance = 75; //tunable
    float height_scale = 4.5;
    cv::Point p1 = cv::Point(frame.size().width/2 - tolerance, frame.size().height/height_scale);        //Top-left point
    cv::Point p2 = cv::Point(0, frame.size().height);                                           //Bottom-left point
    cv::Point p3 = cv::Point(frame.size().width, frame.size().height);                          //Bottom-right point 
    cv::Point p4 = cv::Point(frame.size().width/2 + tolerance, frame.size().height/height_scale);        //Top-right point
    
    //Source points
    cv::Point2f src_points[] = {p1, p2, p3, p4};

    //Define warping points
    int offset = 570; //tunable
    cv::Point pp1 = cv::Point(offset, 0);                                                   //Top-left corner
    cv::Point pp2 = cv::Point(offset, frame.size().height);                                 //Bottom-left corner
    cv::Point pp3 = cv::Point(frame.size().width - offset, frame.size().height);            //Bottom-right corner
    cv::Point pp4 = cv::Point(frame.size().width - offset, 0);                              //Top-right corner

    //Destination points
    cv::Point2f dst_points[] = {pp1, pp2, pp3, pp4};

    //Perspective Transform
    cv::Mat pers_trans = cv::getPerspectiveTransform(src_points, dst_points);

    //Warp raw image
    cv::Mat ipm_frame;
    cv::warpPerspective(frame, ipm_frame, pers_trans, cv::Size(frame.size().width, frame.size().height), cv::INTER_LINEAR);

    return ipm_frame;
}
