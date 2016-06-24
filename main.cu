#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 512
#define GRID_SIZE 256
#define THREADS_NUM (BLOCK_SIZE * GRID_SIZE)

#define SCREEN_WIDTH 1024
#define SCREEN_HEIGHT 648
#define SCREEN_SIZE (SCREEN_WIDTH * SCREEN_HEIGHT)

#define AGENT_NUM 1024
#define AGENT_RADIUS 2

#define rnd() ((float) gpuRand(0.0f, 1.0f))

#define GRAVITY 1e-1f

#define w 0.9
#define a1 0.01
#define a2 0.99
#define dt 0.001

#define CSC(call) {							\
    cudaError err = call;						\
    if(err != cudaSuccess) {						\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));		\
        exit(1);							\
    }									\
} while (0)

GLuint vbo;
struct cudaGraphicsResource *res;

float2 c1, c2; // visible rectangle
float scale = 1;

float *devPos; //[2 * AGENT_NUM];
float *devNewPos; //[2 * AGENT_NUM];
float *devSpeed; //[2 * AGENT_NUM];
float *pbest; //[2 * AGENT_NUM];
float *pmin; //[AGENT_NUM];
float *gbest; //2
float *devCenter;
__device__ float gmin;

// cuda random
__device__ curandState_t globalState[THREADS_NUM];

__global__ void setupCurand(long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &globalState[idx]);
}

__device__ float gpuRand(float a, float b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state = globalState[idx];
    float res = curand_uniform(&state) * (b - a) + a;;
    globalState[idx] = state;
    return res; 
}


__device__ float shvefel(float x, float y) {
	return -x * __sinf(sqrt(abs(x))) - y * __sinf(sqrt(abs(y)));
}

float shv(float x, float y) {
	return -x * sin(sqrt(abs(x))) - y * sin(sqrt(abs(y)));
}
__global__ void putTemperatureField(uchar4* devField, float2 c1, float2 c2) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
    float2 cellSize = make_float2(
        (c2.x - c1.x) / SCREEN_WIDTH,
        (c2.y - c1.y) / SCREEN_HEIGHT
    );
    float minf = -max(abs(c1.x), abs(c2.x)) - max(abs(c1.y), abs(c2.y)); 
    float maxf = max(abs(c1.x), abs(c2.x)) + max(abs(c1.y), abs(c2.y)); 
	for(int i = idx; i < SCREEN_SIZE; i += offsetx) {
        int2 cellCnt = make_int2(i % SCREEN_WIDTH, i / SCREEN_WIDTH);
        float f = shvefel(c1.x +  cellSize.x * cellCnt.x, c1.y + cellSize.y * cellCnt.y);
        unsigned char green = (unsigned char)(255.0 * (f - minf) / (maxf - minf));
        devField[i] = make_uchar4(0, green, 0, 255);
    }
}

__global__ void newCenter(float *devNewPos, float *devCenter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 0) {
        return;
    }
    float x = 0, y = 0;
    for (int i = 0; i < AGENT_NUM; i++) {
        x += devNewPos[2 * i + 0];
        y += devNewPos[2 * i + 1];
    }
    devCenter[0] = x / AGENT_NUM;
    devCenter[1] = y / AGENT_NUM;
}


void createVisibleRectangle() {
    float center[2];
    newCenter<<<GRID_SIZE, BLOCK_SIZE>>>(devNewPos, devCenter);
    cudaMemcpy(center, devCenter, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    float x = center[0];
    float y = center[1];
    c1 = make_float2(x - 0.5 * scale * SCREEN_WIDTH, y - 0.5 * scale * SCREEN_HEIGHT);
    c2 = make_float2(x + 0.5 * scale * SCREEN_WIDTH, y + 0.5 * scale * SCREEN_HEIGHT);
}

__global__ void calcAgentPos(float *devPos, float *devNewPos, float *devSpeed, float *pbest, float *pmin, float *gbest) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENT_NUM) {
        return;
    }
    float2 pos = make_float2(devPos[2 * idx  + 0], devPos[2 * idx + 1]);
    float2 v = make_float2(devSpeed[2 * idx + 0], devSpeed[2 * idx + 1]);

    float rnd1 = gpuRand(0.0, 1.0);
    float rnd2 = gpuRand(0.0, 1.0);

    v.x = w * v.x + 
            (a1  * rnd1 * (pbest[2 * idx + 0] - pos.x)) +
            (a2  * rnd2 * (gbest[0] - pos.x));
    v.y = w * v.y + 
            (a1  * rnd1 * (pbest[2 * idx + 1] - pos.y)) +
            (a2  * rnd2 * (gbest[1] - pos.y));


    for (int i = 0; i < AGENT_NUM; i++) {
        if (idx == i) continue;
        float dx = devPos[2 * i + 0] - devPos[2 * idx + 0];
        float dy = devPos[2 * i + 1] - devPos[2 * idx + 1];
        float s = sqrt(dx * dx + dy * dy);
        v.x -= GRAVITY * dx / (s * s * s * s + 1e-3);
        v.y -= GRAVITY * dy / (s * s * s * s + 1e-3);
    }

    pos.x += v.x * dt;
    pos.y += v.y * dt;

    float f = shvefel(pos.x, pos.y);
    if (f < pmin[idx]) {
        pmin[idx] = f;
        pbest[2 * idx + 0] = pos.x;
        pbest[2 * idx + 1] = pos.y;
    }

    devSpeed[2 * idx + 0] = v.x;
    devSpeed[2 * idx + 1] = v.y;
    devNewPos[2 * idx + 0] = pos.x;
    devNewPos[2 * idx + 1] = pos.y;
}

__global__ void findBest(float *pbest, float *pmin, float *gbest) {
/*
    __shared__ float buf[AGENT_NUM];
    __shared__ float pos[AGENT_NUM * 2];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    buf[idx] = pmin[idx];
    pos[2 * idx] = pbest[2 * idx];
    pos[2 * idx] = pbest[2 * idx + 1];
    if 
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s && buf[idx] < buf[idx + s]) {
            buf[idx] = buf[idx + s];
            pos[2 * idx] = pos[2 * (idx + s)];
            pos[2 * idx + 1] = pos[2 * (idx + s) + 1];
        }
        __syncthreads();
    }
    gmin = buf[0];
    gbest[0] = pos[0];
    gbest[1] = pos[1];
*/
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > 0) {
        return;
    }
    for (int i = 0; i < AGENT_NUM; i++) {
        if (pmin[i] < gmin) {
            gmin = pmin[i];
            gbest[2 * i] = pbest[2 * i];
            gbest[2 * i + 1] = pbest[2 * i + 1];
        }
    }
 
}

__device__ void putAgent(uchar4* devField, int pi, int pj) {
    int i, j;
    j = pj;
    if (0 <= j && j < SCREEN_HEIGHT) {
        for (i = pi - AGENT_RADIUS; i <= pi + AGENT_RADIUS; i++) {
            if (0 <= i && i < SCREEN_WIDTH) {
                devField[j * SCREEN_WIDTH + i] = make_uchar4(255, 0, 0, 255);
            }    
        }
    }
    i = pi;
    if (0 <= i && i < SCREEN_WIDTH) {
        for (j = pj - AGENT_RADIUS; j <= pj + AGENT_RADIUS; j++) {
            if (0 <= j && j < SCREEN_HEIGHT) {
                devField[j * SCREEN_WIDTH + i] = make_uchar4(255, 0, 0, 255);
            }    
        }
    }

}


__global__ void putAgents(uchar4* devField, float *devNewPos, float2 c1, float2 c2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENT_NUM) {
        return;
    }
    float2 cell = make_float2(
        (c2.x - c1.x) / SCREEN_WIDTH,
        (c2.y - c1.y) / SCREEN_HEIGHT
    );
    int i = floor((devNewPos[2 * idx + 0] - c1.x) / cell.x);
    int j = floor((devNewPos[2 * idx + 1] - c1.y) / cell.y);
    putAgent(devField, i, j);
}

void print1() {
    float tmp[2 * AGENT_NUM];
/*
    cudaMemcpy(tmp, devPos, sizeof(float) * 2 * AGENT_NUM, cudaMemcpyDeviceToHost);
    for (int i = 0; i < AGENT_NUM; i++) {
        printf("devPos %d : %f %f\n", i, tmp[2 * i], tmp[2 * i + 1]);
    }
    cudaMemcpy(tmp, pbest, sizeof(float) * 2 * AGENT_NUM, cudaMemcpyDeviceToHost);
    for (int i = 0; i < AGENT_NUM; i++) {
        printf("pbest %d %f %f\n", i, tmp[2 * i], tmp[2 * i + 1]);
    }
*/
    cudaMemcpy(tmp, gbest, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    printf("gbest shvefel(%f, %f) = %f\n", tmp[0], tmp[1],shv(tmp[0], tmp[1]));

}


void print() {
    float tmp[2 * AGENT_NUM];
    cudaMemcpy(tmp, pbest, sizeof(float) * 2 * AGENT_NUM, cudaMemcpyDeviceToHost);
    for (int i = 0; i < AGENT_NUM; i++) {
        printf("pbest %d : %f %f\n", i, tmp[2 * i], tmp[2 * i + 1]);
    }
    cudaMemcpy(tmp, gbest, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    printf("gbest  : %f %f\n",  tmp[0], tmp[1]);
    
//    printf("{%f %f} {%f %f}\n", c1.x, c1.y, c2.x, c2.y);
}

void keyboard(int key, int x, int y) {
    printf("key %d\n", key);
    switch(key) {
        case GLUT_KEY_UP:
                scale = max(scale - 0.01, 0.01f);
                break;
        case GLUT_KEY_DOWN:
                scale = min(scale + 0.01, 10.0f);
                break;
        default:
                break;
    }
}

void update() {
    uchar4 *devField;
    size_t size;
    CSC(cudaGraphicsMapResources(1, &res, 0));
    CSC(cudaGraphicsResourceGetMappedPointer((void**) &devField, &size, res));
   
    printf("===================================\n");
    cudaMemcpy(devPos, devNewPos, sizeof(float) * AGENT_NUM * 2, cudaMemcpyDeviceToDevice); 
    calcAgentPos<<<GRID_SIZE, BLOCK_SIZE>>>(devPos, devNewPos, devSpeed, pbest, pmin, gbest);
    findBest<<<GRID_SIZE, BLOCK_SIZE>>>(pbest, pmin, gbest);
    createVisibleRectangle();
    putTemperatureField<<<GRID_SIZE, BLOCK_SIZE>>>(devField, c1, c2);
    putAgents<<<GRID_SIZE, BLOCK_SIZE>>>(devField, devNewPos, c1, c2);
    print1();

    CSC(cudaDeviceSynchronize());
    CSC(cudaGraphicsUnmapResources(1, &res, 0));
	glutPostRedisplay();
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	glutSwapBuffers();
}

void initGlut(int argc, char **argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutCreateWindow("Andresov Vladislav, group 80-406");
	glutIdleFunc(update);
	glutDisplayFunc(display);
    glutSpecialFunc(keyboard);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble) SCREEN_WIDTH, 0.0, (GLdouble) SCREEN_HEIGHT);

	glewInit();
}

void createVBO() {
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, SCREEN_SIZE * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));
}

void deleteVBO() {
	CSC(cudaGraphicsUnregisterResource(res));
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
}

__global__ void initAgents(float *devNewPos, float *devSpeed, float *pbest, float2 c1, float2 c2) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= AGENT_NUM) {
        return;
    }
    float x = gpuRand(c1.x, c2.x);
    float y = gpuRand(c1.y, c2.y);
    devNewPos[2 * idx + 0] = x;
    devNewPos[2 * idx + 1] = y;
    pbest[2 * idx + 0] = x; 
    pbest[2 * idx + 1] = y;
    devSpeed[2 * idx + 0] = 0;
    devSpeed[2 * idx + 1] = 0;
}

void entry() {
    c1 = make_float2(-0.5 * SCREEN_WIDTH, -0.5 * SCREEN_HEIGHT);
    c2 = make_float2(0.5 * SCREEN_WIDTH, 0.5 * SCREEN_HEIGHT);
    cudaMalloc(&devPos, sizeof(float) * 2 * AGENT_NUM);
    cudaMalloc(&devNewPos, sizeof(float) * 2 * AGENT_NUM);
    cudaMalloc(&devSpeed, sizeof(float) * 2 * AGENT_NUM);
    cudaMalloc(&pbest, sizeof(float) * 2 * AGENT_NUM);
    cudaMalloc(&pmin, sizeof(float) * 2 * AGENT_NUM);
    cudaMalloc(&gbest, sizeof(float) * 2);
    cudaMalloc(&devCenter, sizeof(float) * 2);
    setupCurand<<<GRID_SIZE, BLOCK_SIZE>>>(time(0));
    initAgents<<<GRID_SIZE, BLOCK_SIZE>>>(devNewPos, devSpeed, pbest, c1, c2);
    findBest<<<GRID_SIZE, BLOCK_SIZE>>>(pbest, pmin, gbest);
    print();
}

void leave() {
    cudaFree(devPos);
    cudaFree(devNewPos);
    cudaFree(devSpeed);
    cudaFree(pbest);
    cudaFree(pmin);
    cudaFree(gbest);
    cudaFree(devCenter);
}

int main(int argc, char** argv) {
    initGlut(argc, argv);
    createVBO();
    entry();
	glutMainLoop();
    leave();
    deleteVBO();
	return 0;
}
