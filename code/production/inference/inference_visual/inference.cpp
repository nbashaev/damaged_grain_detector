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

    Mat img, blended;
    namedWindow("initial", WINDOW_AUTOSIZE); moveWindow("initial", 200, 200);
    namedWindow("segm", WINDOW_AUTOSIZE); moveWindow("segm", 200, 200);

    long long ms = 0, cap_ms = 0;
    double components = 0;
    int add;
    int N = 0;
    
    for (auto& p: fs::directory_iterator("../../../data/raw2"))
    {
        auto t_start = chrono::high_resolution_clock::now();
	
        if (p.path().stem().string().back() == 'v')
	    continue;

	img = imread(p.path().string(), CV_LOAD_IMAGE_COLOR);
        blended = img.clone();
        resize(img, img, performer.displaySize, INTER_LINEAR);

	performer.performInference(blended, add);
        components += add;
        N++;
        cout << components / N << endl;

        auto t_end = chrono::high_resolution_clock::now();
        ms += chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
	
        imshow("initial", img);
        imshow("segm", blended);
	
	if (waitKey(0) == 'q')
	    break;
    }


    cout << "avg inference time " << 1e-6 * performer.i_ms / N << " ms" << endl;
    cout << "avg run time " << 1e-6 * ms / N << " ms" << endl;
    cout << endl;


    return 0;
}
