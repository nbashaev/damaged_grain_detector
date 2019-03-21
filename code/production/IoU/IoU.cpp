#include "../InferencePerformer.h"
#include <iostream>


using namespace cv;
using namespace std;
namespace fs = boost::filesystem;


int main(int argc, char *argv[])
{
    string modelName1(argv[1]);
    string modelName2(argv[2]);
    InferencePerformer performer1(modelName1);
    InferencePerformer performer2(modelName2);

    Mat img;

    double IoU = 0.0, I, U;
    double add;

    int N = 0;
    
    for (auto& p: fs::directory_iterator("../../data/raw2"))
    {
	if (p.path().stem().string().back() == 'v')
	    continue;

	img = imread(p.path().string(), CV_LOAD_IMAGE_COLOR);
	

	Mat mask1 = performer1.getMask(img, false) > 0;
	Mat mask2 = performer2.getMask(img, false) > 0;

	I = sum(mask1 & mask2)[0];
	U = sum(mask1 | mask2)[0];
	
	
        if (U > 0)
	    add = (U > 0.0) ? (I / U) : 1.0;

        cout << add << endl;
	IoU += add;

        N++;
    }

    IoU /= N;

    cout << endl << N << ' ' << IoU << endl;


    return 0;
}
