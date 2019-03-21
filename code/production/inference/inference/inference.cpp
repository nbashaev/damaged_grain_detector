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

    Mat img;
    namedWindow("segm", 1);

    long long ms = 0, cap_ms;
    int N = 0;
    Mat img1 = imread("../../../data/raw3/2018-11-20_133328_led.bmp", CV_LOAD_IMAGE_COLOR);
    

    auto t_start = chrono::high_resolution_clock::now();
    
    for (; N < 500; N++)
    {
        img = img1.clone();
	crop(img, alpha, beta);
	performer.performInference(img);
    }

    auto t_end = chrono::high_resolution_clock::now();
    ms += chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

    cout << "avg inference time " << 1e-6 * performer.i_ms / N << " ms" << endl;
    cout << "avg run time " << 1e-6 * ms / N << " ms" << endl;
    cout << endl;


    return 0;
}
