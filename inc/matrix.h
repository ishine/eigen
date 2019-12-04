#pragma once
#include "Utility.h"

Matrix& exp(Matrix &x);
Vector& exp(Vector &x);

Matrix& hard_sigmoid(Matrix &x);
Vector& hard_sigmoid(Vector &x);

Matrix& logistic(Matrix &x);
Vector& logistic(Vector &x);

Matrix& tanh(Matrix &x);
Vector& tanh(Vector &x);

Matrix& relu(Matrix &x);
Vector& relu(Vector &x);

Matrix& softmax(Matrix &x);
Vector& softmax(Vector &x);

Matrix& l2_normalize(Matrix &f);
Vector& l2_normalize(Vector &f);

Matrix& inverse(Matrix &x);
Vector& inverse(Vector &x);

typedef Vector& (*VectorActivator)(Vector &x);
typedef Matrix& (*MatrixActivator)(Matrix &x);

MatrixI& not_equal(MatrixI &x, int y);
MatrixI& equal(MatrixI &x, int y);

MatrixI& operator -=(MatrixI &x, int y);
MatrixI& operator -(int x, MatrixI &y);

vector<Vector>& mean(const vector<Matrix> &x);

vector<Matrix>& sqrt(vector<Matrix> &x);

vector<Vector>& sqrt(vector<Vector> &x);

vector<Matrix>& square(vector<Matrix> &x);

vector<Matrix>& operator -(vector<Matrix> &x, const vector<Matrix> &y);

vector<Matrix>& operator /(vector<Matrix> &x, const vector<Matrix> &y);

vector<Matrix>& operator /(vector<Matrix> &x, const vector<Vector> &y);

vector<Matrix>& operator +(vector<Matrix> &x, double y);

vector<Vector>& operator +(vector<Vector> &x, double y);

vector<Matrix>& operator *(vector<Matrix> &x, const Vector &y);

vector<Matrix>& operator +(vector<Matrix> &x, const Vector &y);

vector<Matrix>& operator -(vector<Matrix> &x, const vector<Vector> &y);
