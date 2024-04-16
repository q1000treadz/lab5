#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;
const int WIDTH = 800;
const int HEIGHT = 800;
const int MAX_ITERATIONS = 2000;

int mandelbrot(const complex<double>& c, int max_iterations) {
    complex<double> z = 0;
    int iterations = 0;
    while (abs(z) < 3 && iterations < max_iterations) {
        z = z * z + c;
        iterations++;
    }
    return iterations;
}

void calculateMandelbrot(int* data, int rank, int rows_per_process, int width, int height) {
    for (int row = rank * rows_per_process; row < (rank + 1) * rows_per_process; ++row) {
        for (int col = 0; col < width; ++col) {
            double x = (col - width / 2.0) * 4.0 / width;
            double y = (row - height / 2.0) * 4.0 / height;

            complex<double> c(x, y);
            data[(row - rank * rows_per_process) * width + col] = mandelbrot(c, MAX_ITERATIONS);
        }
    }
}

void createImage(int* recv_buffer, int width, int height) {
    Mat image(height, width, CV_8UC3);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int iterations = recv_buffer[row * width + col];
            if (iterations == MAX_ITERATIONS) {
                image.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
            }
            else {
                image.at<Vec3b>(row, col) = Vec3b(255, (iterations % 17 + 1) / 2.0 * 255, (iterations % 3 + 1) / 5.5 * 255);;
            }
        }
    }
    imshow("Mandelbrot", image);
    waitKey(0);
}

int main() {
    

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows_per_process = HEIGHT / size;

    int* data = new int[WIDTH * rows_per_process];

    calculateMandelbrot(data, rank, rows_per_process, WIDTH, HEIGHT);

    int* recv_buffer = NULL;
    if (rank == 0) {
        recv_buffer = new int[WIDTH * HEIGHT];
    }

    MPI_Gather(data, WIDTH * rows_per_process, MPI_INT, recv_buffer, WIDTH * rows_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        createImage(recv_buffer, WIDTH, HEIGHT);
    }

    delete[] data;
    if (rank == 0) {
        delete[] recv_buffer;
    }

    MPI_Finalize();

    return 0;
}