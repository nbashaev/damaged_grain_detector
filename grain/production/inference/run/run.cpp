#include "../../InferencePerformer.h"
#include <iostream>


using namespace cv;
using namespace std;
namespace fs = boost::filesystem;


int main(int argc, char *argv[])
{
    string modelName(argv[1]);
    InferencePerformer performer(modelName);

    char *junk_ptr;
    double alpha = 1.0;
    double beta = 1.0;

    if (argc == 4)
    {
	alpha = strtod(argv[2], &junk_ptr);
	beta = strtod(argv[3], &junk_ptr);
    }


    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    Mat img;
    namedWindow("segm", 1);

    long long ms = 0, cap_ms = 0;
    int N = 0;

    auto t_start = chrono::high_resolution_clock::now();
    
    for (;; N++)
    {
        auto tt_start = chrono::high_resolution_clock::now();

        cap >> img;
        crop(img, alpha, beta);

        auto tt_end = chrono::high_resolution_clock::now();
        cap_ms += chrono::duration_cast<std::chrono::nanoseconds>(tt_end - tt_start).count();

	performer.performInference(img);
	imshow("segm", img);
	
	if (waitKey(1) >= 0)
	    break;
    }

    auto t_end = chrono::high_resolution_clock::now();
    ms += chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

    cout << "avg cap time " << 1e-6 * cap_ms / N << " ms" << endl;
    cout << "avg inference time " << 1e-6 * performer.i_ms / N << " ms" << endl;
    cout << "avg run time " << 1e-6 * ms / N << " ms" << endl;
    cout << endl;


    return 0;
}
