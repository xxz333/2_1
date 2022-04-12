#include<xmmintrin.h> //SSE
#include<emmintrin.h> //SSE2
#include<pmmintrin.h> //SSE3
#include<tmmintrin.h> //SSSE3
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h> //AVX、AVX2
#include<iostream>
#include<time.h>

using namespace std;
//struct timeval val;
//struct timeval newval;

//测试用例生成
void m_reset(float **m,int N)
{
        for(int i=0;i<N;i++)
        {
                for(int j=0;j<i;j++)
                        m[i][j]=0;
                m[i][i]=1.0;
                for(int j=i+1;j<N;j++)
                        m[i][j]=rand();
        }
        for(int k=0;k<N;k++)
                for(int i=k+1;i<N;i++)
                        for(int j=0;j<N;j++)
                                m[i][j]+=m[k][j];
}

//普通高斯消元
void gauss(float **A,int n)
{
        for(int k=0;k<n;k++)
        {
                //vt ← dupToFloat(A[k,k]);
                __m128 vt=_mm_set1_ps(A[k][k]);
                int j;
                for(j=k+1;j+4<=n;j+=4)
                {
                      	if(j%4!=0)
			{
				do
				{
					A[k][j]=A[k][j]/A[k][k];
					j++;
				}
				while(j%4!=0&&j<n);
			}
			if(j+4<=n)
			{
                       		__m128 va=_mm_load_ps(&A[k][j]);
                       		//向量对位相除
                        	//va<-va/vt;
                        	va=_mm_div_ps(va,vt);
                        	//将四个单精度浮点数从向量寄存器存储到内存
				_mm_store_ps(&A[k][j],va);
			}
                }
                for(;j<n;j++)
                        A[k][j]=A[k][j]/A[k][k];
                A[k][k]=1.0;
                for(int i=k+1;i<n;i++)
                {
                        __m128 vaik=_mm_set1_ps(A[i][k]);
                        for(j=k+1;j+4<=n;j+=4)
                        {
                                __m128 vakj=_mm_load_ps(&A[k][j]);
                                __m128 vaij=_mm_load_ps(&A[i][j]);
                                //vx ← vakj*vaik;
                                __m128 vx=_mm_mul_ps(vakj,vaij);
                                vaij=_mm_sub_ps(vaij,vx);
                                //store4FloatTo(&A[i,j],vaij);
                                _mm_store_ps(&A[i][j],vaij);
                        }
                        for(;j<n;j++)
                                A[i][j]=A[i][j]-A[k][j]*A[i][j];
                        A[i][k]=0;
                 }
        }
}


int main()
{
        int N;
	cin>>N;
	float **m;
        for(int i=0;i<N;i++)
                m=new float*[N];
        for(int i=0;i<N;i++)
                m[i]=new float[N];
        int n;//执行次数
        m_reset(m,N);//测试用例生成
        clock_t start,finish;
        start=clock();
        for(int i=0;i<n;i++)
           gauss(m,N);
	finish=clock();
        cout<<n<<":"<<(finish-start)/float(CLOCKS_PER_SEC)<<endl;
        return 0;
}

