//This file is Vishal's personal iris test as a checker, and decision tree verifier

#include <stdio.h>
#include <iostream>
#include "tree.cuh"
#include <fstream>
#include <sstream>
#include <string>

#define N 150

using namespace std;
int main()
{
	ifstream myfile;
	myfile.open("data.csv");
	string line;
	vector<float> data;
	vector<int> labels;
	int counter = 0;
	data.resize(N*4);
	labels.resize(N);
	
	while(getline(myfile,line))
		{
			stringstream str(line);
			vector<float> row;
			float i;
			while ( str >> i)
				{
					row.push_back(i);
					if(str.peek() == ',')
						str.ignore();
				}
			data[counter + 0*N] = row[0];
			data[counter + 1*N] = row[1];
			data[counter + 2*N] = row[2];
			data[counter + 3*N] = row[3];
			labels[counter] = (int)row[4];
			counter++;
		}
	cout << "Lines processed " << counter << endl;  
	myfile.close();
	
	float *d_data;
	int *d_labels;
	unsigned int* rowids;
	unsigned int* h_rowids = (unsigned int*)malloc(N*sizeof(int));
	
	for(int i=0;i<N;i++)
		{
			h_rowids[i] = i;
		}
	
	cudaMalloc((void**)(&d_data),N*4*sizeof(float));
	cudaMalloc((void**)(&d_labels),N*sizeof(int));
	cudaMalloc((void**)(&rowids),N*sizeof(int));
	
	cudaMemcpy(d_data,data.data(),N*4*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels,labels.data(),N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(rowids,h_rowids,N*sizeof(unsigned int),cudaMemcpyHostToDevice);

	ML::DecisionTree::DecisionTreeClassifier model;
	model.plant(d_data,4,N,d_labels,rowids,N,8);
	model.print();
	
	cudaFree(d_data);
	cudaFree(d_labels);
	cudaFree(rowids);
	
	return 0;
	
}
