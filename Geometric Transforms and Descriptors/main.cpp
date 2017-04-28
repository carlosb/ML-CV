/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Author: Carlos Brito (carlos.brito524@gmail.com)
 * Date: 3/22/17.
 *
 * Description:
 * This is a small piece of code to demonstrate the power
 * and "descriptiveness" of image moments and fourier des-
 * criptors.
 *
 * TODO:
 * We have to adjust the calculation of the fourier descriptors
 * so we can make it invariant to rotation and scale.
 *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/
//</editor-fold>

#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <climits>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const static double PI = 4*atan(1); // ensures maximum precision on the architecture

typedef vector<Point> Contour; // a contour is a list of points

typedef struct FourierDescriptors {
    size_t n; // size of vector array
    double (*v)[2]; // descriptor vector
    Contour filtered_contour; // filtered contour

    FourierDescriptors (size_t nsize) {
        n = nsize;
        v = new double[n][2];
    }


} FourierDescriptors;

vector<Contour> getContours(Mat const &binary_img); // returns contours of a binary image

void drawContour(Contour const& cont, Mat &dst); // draws a white contour over a black image

void geomTransform(Contour const &contourIn,
                   Contour &contourOut,
                   double scale,
                   double translatex,
                   double translatey,
                   double theta);

Point2d getCentroid(Contour const& contour); // returns the centroid of a contour

void getHuMoments(Contour const& contour, vector<double> &hu); // returns the hu moments of a contour

void getFourierDescriptors(Contour const& contour, FourierDescriptors &fourierDescriptorsInOut); // returns the fourier descriptor data

vector<double> VectorDiff(vector<double> p1, vector<double> p2);
double L2Norm(vector<double> p);

int main(int argc, char **argv) {

    // Get image name
    if(argc != 2) {
        printf("Usage %s imagen\n", *argv);
        exit(1);
    }
    char* image_name = argv[1];

    // Read image
    Mat img;
    img = imread(image_name);
    cvtColor(img, img, CV_RGB2GRAY); // make sure we only get one channel
    imshow("img", img);

    // Threshold image
    Mat thresh;
    threshold(img,thresh, 127, 255, THRESH_BINARY_INV);
    thresh.convertTo(thresh, CV_8UC1);
    imshow("thresh",thresh);

    // Get contour of image
    vector<Contour> contours = getContours(thresh);

    if(contours.size() > 1)
        printf("Warning: More than one contour detected");

    Contour contour = getContours(thresh)[0];
    Contour transformedContour;

    geomTransform(contour, transformedContour, 1.5, 20, 50, PI/3);

    // Visualize contour
    Mat contourImg;
    drawContour(transformedContour, contourImg);
    imshow("transformed contour", contourImg);

    // First we compute Hu moments
    vector<double> hu(7);
    vector<double> transformedHu(7);
    getHuMoments(contour,hu);
    getHuMoments(transformedContour,transformedHu);


    cout << "Hu moments for " << image_name << endl;
    for(vector<double>::iterator moment = hu.begin(); moment != hu.end(); moment++) {
        cout << *moment << endl;
    }


    cout << "\nHu moments for transformed " << image_name << endl;
    for(vector<double>::iterator moment = transformedHu.begin(); moment != transformedHu.end(); moment++) {
        cout << *moment << endl;
    }

    cout << "L2 Norm of difference of Hu moments: " << L2Norm(VectorDiff(hu, transformedHu)) << endl;


    // Second we compute fourier descriptors
    FourierDescriptors FD(15);
    FourierDescriptors transformedFD(15);

    getFourierDescriptors(contour, FD);
    getFourierDescriptors(transformedContour, transformedFD);


    cout << "\nFourier descriptor vector for " << image_name << endl;
    for (int i = 0; i < FD.n; ++i) {
        cout << "( " << FD.v[i][0] << ", " << FD.v[i][1] << " )" << endl;
    }

    cout << "\nFourier descriptor vector for " << image_name << endl;
    for (int i = 0; i < transformedFD.n; ++i) {
        cout << "( " << transformedFD.v[i][0] << ", " << transformedFD.v[i][1] << " )" << endl;
    }

    Mat filtered_image;
    drawContour(transformedFD.filtered_contour, filtered_image);
    imshow("filtered", filtered_image);

    waitKey(0);

    return 0;
}

// ---------------------
// gets hu moments of a contour
void getHuMoments(Contour const& contour, vector<double> &hu) {

    vector<double> hu_array(7);

    Moments mu = moments(contour, true); // we assume there is only one contour we're interested in
    HuMoments(mu, hu_array);
    hu = hu_array;
}

// ---------------------
// gets fourier descriptor data
void getFourierDescriptors(Contour const& contour, FourierDescriptors &fourierDescriptorsInOut) {

    int n = fourierDescriptorsInOut.n;

    // calculate centroid
    Point2d centroid = getCentroid(contour);

    // center and convert to complex plane
    fftw_complex* s = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * contour.size());
    for (unsigned int i = 0; i < contour.size(); ++i) {
        s[i][0] = contour.at(i).x - centroid.x;
        s[i][1] = contour.at(i).y - centroid.y;
    }

    // Perform fft on contour
    fftw_complex* S = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * contour.size());
    fftw_plan fft = fftw_plan_dft_1d(contour.size(), s, S, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(fft);

    // convert to mag-phase representation
    double (*rep)[2] = new double[contour.size()][2];

    for (int i = 0; i < contour.size(); ++i) {
        double real = S[i][0];
        double imag = S[i][1];

        rep[i][0] = sqrt(real*real + imag*imag) / contour.size(); // magnitude
        rep[i][1] = atan2(imag, real); // phase
    }

    // filter out high frequencies
    for (int i = 0; i < contour.size(); ++i) {
        if(i > n && i < (contour.size() - n)) {
            rep[i][0] = 0; // zero out the magnitude
        }
    }

    // convert back to real+imag representation
    fftw_complex *Sfilt = (fftw_complex *) fftw_malloc(contour.size() * sizeof(fftw_complex));
    for (int i = 0; i < contour.size(); ++i) {
        Sfilt[i][0] = rep[i][0] * cos(rep[i][1]);
        Sfilt[i][1] = rep[i][0] * sin(rep[i][1]);

        if (i >= 1 && i < n+1) {
            fourierDescriptorsInOut.v[i-1][0] = rep[i][0];
            fourierDescriptorsInOut.v[i-1][1] = rep[i][1];
        }
    }

    // Perform ifft on filtered contour
    fftw_complex *sfilt = (fftw_complex *) fftw_malloc(contour.size() * sizeof(fftw_complex));
    fftw_plan  ifft = fftw_plan_dft_1d((int) contour.size(), Sfilt, sfilt, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(ifft);

    Contour filtered_contour(contour.size());
    for (int i = 0; i < contour.size(); ++i) {
        filtered_contour[i].x = (int)( sfilt[i][0] + centroid.x);
        filtered_contour[i].y = (int)( sfilt[i][1] + centroid.y );
    }

    fourierDescriptorsInOut.filtered_contour = filtered_contour;

    fftw_free(s);
    fftw_free(S);
    fftw_free(sfilt);
    fftw_free(Sfilt);
    delete[] rep;
}

// ---------------------
// gets contours of binary image
vector<Contour> getContours(Mat const &binary_img) {
    vector<Contour> contours;
    findContours(binary_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    return contours;
}

// ---------------------
// gets centroid of contour
Point2d getCentroid(Contour const& contour) {
    Point2d centroid = Point(0,0);
    for (unsigned int i = 0; i < contour.size(); ++i) {
        centroid += (Point2d) contour.at(i);
    }
    centroid /= double( contour.size() );

    return centroid;
}


// ---------------------
// applies affine transform
void geomTransform(Contour const &contourIn, Contour &contourOut, double scale, double translatex, double translatey,
                   double theta) {

    Contour contour0 = contourIn;
    Point2d centroid = getCentroid(contour0);

    for (int i = 0; i < contour0.size(); ++i) {
        double xi = contour0[i].x;
        double yi = contour0[i].y;
        contour0[i].x = (int) (scale*( cos(theta)*(xi - centroid.x) - sin(theta)*(yi - centroid.y) ) + centroid.x + translatex);
        contour0[i].y = (int) (scale*( sin(theta)*(xi - centroid.x) + cos(theta)*(yi - centroid.y) ) + centroid.y + translatey);
    }

    contourOut = contour0;

}

// -----------------------
// draws a contour of a black image
void drawContour(Contour const& cont, Mat &dst) {

    int max_x = INT_MIN;
    int max_y = INT_MIN;

    int min_x = INT_MAX;
    int min_y = INT_MAX;

    for (int i = 0; i < cont.size(); ++i) {
        // get max value of x and y
        if(max_x < cont[i].x)
            max_x = cont[i].x;
        if(max_y < cont[i].y)
            max_y = cont[i].y;

        // get min value of x and y
        if(min_x > cont[i].x)
            min_x = cont[i].x;
        if(min_y > cont[i].y)
            min_y = cont[i].y;
    }

    Contour shifted_contour;
    geomTransform(cont, shifted_contour, 1, abs(min_x) + 1, abs(min_y) + 1, 0);

    vector<Contour> new_contours;
    new_contours.push_back(shifted_contour);

    Mat m = Mat::zeros(max_y + abs(min_y) + 2, max_x + abs(min_x) + 2, CV_8UC1);
    drawContours(m, new_contours, 0, 255);

    dst = m;
}


vector<double> VectorDiff(vector<double> p1, vector<double> p2) {
    vector<double> result;
    result.reserve(p1.size());

    if(p1.size() != p2.size()) {
        for (int i = 0; i < p1.size(); ++i) {
            result[i] = -1;
        }
    }

    for (int i = 0; i < p1.size(); ++i) {
        result[i] = p1[i] - p2[i];
    }

    return result;
}

double L2Norm(vector<double> p) {
    double result = 0;

    for (int i = 0; i < p.size(); ++i) {
        result += p[i]*p[i];
    }

    result = sqrt(result);

    return result;
}
