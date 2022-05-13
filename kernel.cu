#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <glut.h>
#include <stdio.h>
#include <cmath>



int WinWid = 1280, WinHei = 720;  // Window width and height

int thread_number = 100;  // Number of threads/functions
int point_offset = 10, points_number = 50;  // Distance between points and number of points in one function
int y_offset = 2;  // Distance between functions, Y axis
int time_delay = 20;  // Frame update delay
int dx = 1;  // X change each frame

int* cpu_coordinates;  // Array of point coordinates
int* gpu_coordinates;  // Array of point coordinates, allocated on GPU
float* gpu_coefficients;  // Array of function coefficients, allocated on GPU


// Drawable function
__device__ int f(int x, float c1, float sin_coef, float cos_coef) {
    return int(100 * (sin_coef * sin(x/c1 * 3.14/180) + cos_coef * cos(x/c1 * 3.14/180)));
}

// Multithreading function, called each timer tick
__global__ void getNextPosition(int* coords, float* gpu_coefficients, int thread_num, int points_num, int dx, int y_offset) {
    int t_id = threadIdx.x;
	if(t_id < thread_num) {  // Overflow check
        int thread_offset = t_id * points_num * 2;  // Offset in array between threads
        // Getting values from coefficient array
        float c1 = gpu_coefficients[t_id * 3];
        float sin_coef = gpu_coefficients[t_id * 3 + 1];
        float cos_coef = gpu_coefficients[t_id * 3 + 2];
		for (int i = 0; i < points_num; i++)
        {
            // Calculating new points
            coords[thread_offset + i * 2] += dx;
            coords[thread_offset + i * 2 + 1] = f(coords[thread_offset + i * 2], c1, sin_coef, cos_coef) + t_id * y_offset - (thread_num * y_offset / 2);
        }
        
    }
}

// Redrawing function
void draw() {

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_POINT_SMOOTH);
    glPushMatrix();
    glScalef(1 / ((float)WinWid / 2), 1 / ((float)WinHei / 2), 1);


    for (int i = 0; i < thread_number * points_number; i++) {
    	glColor3f(1, 0, 0);
    	glBegin(GL_POINTS);
        glPointSize(15);
        glVertex2i(cpu_coordinates[i*2] % WinWid - WinWid / 2, cpu_coordinates[i*2+1]);
        glEnd();
    }

    glPopMatrix();
    glutSwapBuffers();

}

// Timer function, called every time_delay msec
void timer(int value) {

    int size = thread_number * points_number * 2;
    cudaMemcpy(gpu_coordinates, cpu_coordinates, size * sizeof(int), cudaMemcpyHostToDevice);  // Copying from CPU
    getNextPosition<<<1, thread_number >>>(gpu_coordinates, gpu_coefficients, thread_number, points_number, dx, y_offset);  // Calculating next coordinates
    cudaThreadSynchronize();  // Synchronizing threads
    cudaMemcpy(cpu_coordinates, gpu_coordinates, size * sizeof(int), cudaMemcpyDeviceToHost);  // Copying back to CPU

    // Calling redrawing
    glutPostRedisplay();  
    glutTimerFunc(time_delay, timer, 0);

}


// Filling our thread/function coefficients with random numbers
void fillCoefficients() {
    int size = thread_number * 3;
    cudaMalloc(&gpu_coefficients, size * sizeof(float));
    float* ptr = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < thread_number; i++) {
        ptr[i * 3] = (float)rand() / (float)RAND_MAX * (5 - 2) + 1;  //[2, 5]
        ptr[i * 3 + 1] = (float)rand() / (float)RAND_MAX;  //[0, 1]
        ptr[i * 3 + 2] = (float)rand() / (float)RAND_MAX;  //[0, 1]
    }
    cudaMemcpy(gpu_coefficients, ptr, size * sizeof(float), cudaMemcpyHostToDevice);
}


// Initializing function
void init() {

    glClearColor(0.0, 0.0, 0.0, 1.0);

    glMatrixMode(GL_PROJECTION);

    srand(time(NULL));
    cpu_coordinates = (int*)malloc(thread_number * points_number * 2 * sizeof(int));
    for (int i = 0; i < thread_number; i++) {
        int thread_offset = i * points_number * 2;
        int random_x_offset = rand() % WinWid;
        for (int j = 0; j < points_number; j++)
            cpu_coordinates[thread_offset + j * 2] = random_x_offset + point_offset * j;
    }
    fillCoefficients();

    int size = thread_number * points_number * 2;
    cudaMalloc(&gpu_coordinates, size * sizeof(int));  // Allocating memory for coordinates array

    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);

}





int main(int argc, char** argv) {

    // Setting up OpenGL window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);


    glutInitWindowSize(WinWid, WinHei);
    glutInitWindowPosition(400, 150);
    glutCreateWindow("SIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIINUS");


    glutDisplayFunc(draw);
    glutTimerFunc(60, timer, 0);

    init();

    glutMainLoop();

    // Releasing allocated memory
    free(cpu_coordinates);
    cudaFree(gpu_coordinates);  
    cudaFree(gpu_coefficients);

}