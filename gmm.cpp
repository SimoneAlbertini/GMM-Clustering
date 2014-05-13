/* 
 * File:   gmm.cpp
 * Author: Simone Albertini
 * 
 * E-mail: albertini.simone@gmail.com
 */

#include <iostream>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::vector;


bool is_black(cv::Vec3b c1)
{
    return c1[0] == 0 && c1[1] == 0 && c1[2] == 0;
}

std::vector<cv::Vec3b> get_colors(const int n)
{
    vector<cv::Vec3b> vec;
    vec.push_back(cv::Vec3b(255,0,0));
    vec.push_back(cv::Vec3b(0,255,0));
    vec.push_back(cv::Vec3b(0,0,255));
    vec.push_back(cv::Vec3b(0,255,255));
    vec.push_back(cv::Vec3b(255,255,0));
    vec.push_back(cv::Vec3b(255,0,255));
    
    for(int i=6; i < n; i++)
    {
        cv::Vec3b color(cv::saturate_cast<uchar>(rand()*255),
                        cv::saturate_cast<uchar>(rand()*255),
                        cv::saturate_cast<uchar>(rand()*255));
        vec.push_back(color);
    }
    
    return vec;
}

int main(int argc, char** argv) 
{
    string filename = "data.png";
    int num_clusters = 15;
    
    if(argc > 1 && string(argv[1]) == "help") 
    { 
        cout << "Usage: gmm [data file] [number clusters]" << endl;
        return EXIT_SUCCESS;
    }
    
    if(argc > 1) filename = string(argv[1]);
    if(argc > 2) num_clusters = atoi(argv[2]);
    
    cv::Mat data = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    
    cv::Mat patterns(0,0, CV_32F);
    
    for(int r=0; r < data.rows; r++)
        for(int c=0; c < data.cols; c++)
        {
            cv::Vec3b p = data.at<cv::Vec3b>(r,c);
            if(!is_black(p))
            {
                cv::Mat pat = (cv::Mat_<float>(1,2) << c,r);
                patterns.push_back(pat);
            }
            
        }
    
    const int cov_mat_type = cv::EM::COV_MAT_GENERIC;
    cv::TermCriteria term(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 15000, 1e-6);
    
    cv::EM gmm(num_clusters, cov_mat_type, term);
    
    cout << "num samples: " << patterns.rows << endl;
    
    cv::Mat labels, posterior, logLikelihood;
    
    cout << "Training GMM... " << flush;
    gmm.train(patterns, logLikelihood, labels, posterior);
    cout << "Done!" << endl;
    
    
    
    cv::Mat display_img(data.size(), CV_8UC3);
    cv::Mat posterior_img(data.size(), CV_8U);
    
    vector<cv::Vec3b> colors = get_colors(num_clusters);
    
    // Draw points with labels
    for(int p=0; p < patterns.rows; p++)
    {
        cv::Mat pat = patterns.row(p);
        int r = pat.at<float>(0,1);
        int c = pat.at<float>(0,0);
        
        display_img.at<cv::Vec3b>(r,c) = colors[labels.at<int>(p,0)];
        posterior_img.at<uchar>(r,c) = cv::saturate_cast<uchar>(posterior.at<float>(r,c)*255);
    }
    
    // draw components
    cv::Mat means = gmm.get<cv::Mat>("means");
    vector<cv::Mat> covs = gmm.get<vector<cv::Mat> >("covs");
    cv::Mat weights = gmm.get<cv::Mat>("weights");
    
    for(int g=0; g < num_clusters; g++)
    {
        double cx = means.at<double>(g, 0);
        double cy = means.at<double>(g, 1);
        double w = weights.at<double>(0, g);
        cv::Mat cov = covs[g];
        
        // draw centroid
        cv::circle(display_img, cv::Point(cx,cy), 2, cv::Scalar(255,255,255), CV_FILLED);
        
        // draw eigenvectors
        cv::Mat eigVal, eigVec;
        cv::eigen(cov, eigVal, eigVec);
        
        double eigVec1_len = sqrt(eigVal.at<double>(0,0)) * 3;
        double eigVec1_x = eigVec.at<double>(0,0) * eigVec1_len;
        double eigVec1_y = eigVec.at<double>(0,1) * eigVec1_len;
        double eigVec2_len = sqrt(eigVal.at<double>(1,0)) * 3;
        double eigVec2_x = eigVec.at<double>(1,0) * eigVec2_len;
        double eigVec2_y = eigVec.at<double>(1,1) * eigVec2_len;
        
        cv::line(display_img, cv::Point(cx,cy), cv::Point(cx+eigVec1_x, cy+eigVec1_y), cv::Scalar(255,255,255) );
        cv::line(display_img, cv::Point(cx,cy), cv::Point(cx+eigVec2_x, cy+eigVec2_y), cv::Scalar(255,255,255) );
        
        // draw ellipse along eigenvector 1
        double angle = atan(eigVec1_y / eigVec1_x) * (180 / M_PI);
        cv::RotatedRect rect(cv::Point(cx, cy), cv::Size(eigVec1_len, eigVec2_len), angle);
        double min, max; cv::minMaxLoc(weights, &min, &max);
        uchar intensity = cv::saturate_cast<uchar>(w * 255.0 / max);
        cv::ellipse(display_img, rect, cv::Scalar(intensity,intensity,intensity), 1);
        
    }
    
    cv::namedWindow("display");
    cv::namedWindow("posterior");
    cv::imshow("display", display_img);
    cv::imshow("posterior", posterior_img);
    
    cv::waitKey();
    
    return EXIT_SUCCESS;
}

