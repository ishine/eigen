#include <iostream>
#include <ctime>
#include <Windows.h>
using namespace std;
#include "Strassen.h"

int test_Strassen()
{
    Strassen_class<int> stra;//定义Strassen_class类对象
    int MatrixSize = 0;

    int** MatrixA;    //存放矩阵A
    int** MatrixB;    //存放矩阵B
    int** MatrixC;    //存放结果矩阵

    clock_t startTime_For_Normal_Multipilication ;
    clock_t endTime_For_Normal_Multipilication ;

    clock_t startTime_For_Strassen ;
    clock_t endTime_For_Strassen ;
    srand(time(0));

    cout<<"\n请输入矩阵大小(必须是2的幂指数值(例如:32,64,512,..): ";
    cin>>MatrixSize;

    int N = MatrixSize;//for readiblity.

    //申请内存
    MatrixA = new int *[MatrixSize];
    MatrixB = new int *[MatrixSize];
    MatrixC = new int *[MatrixSize];

    for (int i = 0; i < MatrixSize; i++)
    {
        MatrixA[i] = new int [MatrixSize];
        MatrixB[i] = new int [MatrixSize];
        MatrixC[i] = new int [MatrixSize];
    }

    stra.FillMatrix(MatrixA,MatrixB,MatrixSize);  //矩阵赋值

  //*******************conventional multiplication test
        cout<<"朴素矩阵算法开始时钟:  "<< (startTime_For_Normal_Multipilication = clock());

        stra.MUL(MatrixA,MatrixB,MatrixC,MatrixSize);//朴素矩阵相乘算法 T(n) = O(n^3)

        cout<<"\n朴素矩阵算法结束时钟: "<< (endTime_For_Normal_Multipilication = clock());

        cout<<"\n矩阵运算结果... \n";
        stra.PrintMatrix(MatrixC,MatrixSize);

  //*******************Strassen multiplication test
        cout<<"\nStrassen算法开始时钟: "<< (startTime_For_Strassen = clock());

        stra.Strassen( N, MatrixA, MatrixB, MatrixC ); //strassen矩阵相乘算法

        cout<<"\nStrassen算法结束时钟: "<<(endTime_For_Strassen = clock());


    cout<<"\n矩阵运算结果... \n";
    stra.PrintMatrix(MatrixC,MatrixSize);

    cout<<"矩阵大小 "<<MatrixSize;
    cout<<"\n朴素矩阵算法: "<<(endTime_For_Normal_Multipilication - startTime_For_Normal_Multipilication)<<" Clocks.."<<(endTime_For_Normal_Multipilication - startTime_For_Normal_Multipilication)/CLOCKS_PER_SEC<<" Sec";
    cout<<"\nStrassen算法:"<<(endTime_For_Strassen - startTime_For_Strassen)<<" Clocks.."<<(endTime_For_Strassen - startTime_For_Strassen)/CLOCKS_PER_SEC<<" Sec\n";
    system("Pause");
    return 0;

}
