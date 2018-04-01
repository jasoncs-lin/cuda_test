
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
//--------------------------------------------------------------------
//A : m x l
//B : l x n
//C : m x n    (C=A*B)
//--------------------------------------------------------------------
 
void host_mm(float* C, float* A, float* B, int m, int n, int l){
 
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++) 
    {
            float s=0;
            for (int k=0; k<l; k++)
        {
                float a = A[i*l + k];
                float b = B[k*n + j];
                s += a * b;
            }
            C[i*n + j] = s;
        }
}
 
//--------------------------------------------------------------------
__global__ void gpu_mm(float* C, float* A, float* B, int m, int n, int l){
    
    //// 2D Thread ID    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Pvalue is used to store the element of the matrix
    // that is computed by the thread    
    float Pvalue = 0;
    
    for (int k = 0; k < l; ++k)
    { 
         float Aelement = A[ty * l + k];
         float Belement = B[k * n + tx];
         Pvalue += Aelement * Belement;
    }
    // Write the matrix to device memory;
    // each thread writes one element
    C[ty * n + tx] = Pvalue;
    //printf("threadIdx.x=%d\n", threadIdx.x);
}
 
 
//----------------------------------------------
double diff(float* a, float* b, int n){
    double s=0, r=0;
    for(int k=0; k<n; k++)
    {
        double w=a[k]-b[k];
        s+=w*w;
        r+=a[k]*a[k];
    }
    return sqrt(s/r); 
}
 
 
void random(float* a, int n){
    for(int k=0; k<n; k++){
        a[k]=(float)rand()/RAND_MAX*2-1;
    }
}
 
//----------------------------------------------
void testMatrix(int m, int n, int l)
{
    //initialize
    float *a = (float*)malloc(sizeof(float)*m*l);
    float *b = (float*)malloc(sizeof(float)*l*n);
    float *c1 = (float*)malloc(sizeof(float)*m*n);
    float *c2 = (float*)malloc(sizeof(float)*m*n);
 
    srand(time(0));
    random(a,m*l);
    random(b,l*n);
    memset(c1, 0, sizeof(float)*m*n);
    memset(c2, 0, sizeof(float)*m*n);
        
    float  *ga,*gb,*gc;
    cudaMalloc((void**)&ga, m*l*sizeof(float));
    cudaMalloc((void**)&gb, l*n*sizeof(float));
    cudaMalloc((void**)&gc, m*n*sizeof(float));
 
    cudaMemcpy(ga, a, m*l*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gb, b, l*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gc, 0, m*n*sizeof(float));
        
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); 
 
    //SBMT(Single Block, Multiple Threads)
    gpu_mm<<<dim3(1,1,1), dim3(m, n, 1)>>> (gc,ga,gb,m,n,l);
    cudaThreadSynchronize();
 
    cudaEventRecord(stop,0); 
    cudaEventSynchronize(stop);
 
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cudaMemcpy(c2, gc, m*n*sizeof(float), cudaMemcpyDeviceToHost);        
    
    double c_start,c_stop;
    double CPU_execution_time;
    c_start = (double)clock();
 
    host_mm(c1, a, b, m, n, l);
     
    c_stop = (double)clock();
    CPU_execution_time = (c_stop - c_start)/(double)CLOCKS_PER_SEC;
 
    //check precision        
    double err=diff(c1,c2,m*n);
    printf("err = %g\n", err);       
 
    printf(" ======== (Execution Infomation) ========\n");
    printf(" Excuetion Time on GPU: %3.20f s\n",elapsedTime/1000);
    printf(" Excuetion Time on CPU: %3.20f s\n",CPU_execution_time);
    printf(" Speed up = %f\n",(CPU_execution_time/(elapsedTime/1000)));
    printf(" ========================================\n\n");
 
 
    free(a); 
    free(b); 
    free(c1); 
    free(c2); 
       
    cudaFree(ga);
    cudaFree(gb);
    cudaFree(gc);
        
}
 
//----------------------------------------------
int main()
{
 
    int m=32;
    int n=32;
    int l=32;
 
    testMatrix(m,n,l);
 
    return 0;
}
