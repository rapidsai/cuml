//This file is Vishal's personal iris test as a checker, and decision tree verifier

#include <stdio.h>
#include <iostream>
#include "tree.cuh"
#include <fstream>
#include <sstream>
#include <string>

#define N 150
#define COLS 4
#define depth -1
using namespace std;
int main()
{
	ifstream myfile;
	myfile.open("data.csv");
	string line;
	vector<float> data;
	vector<int> labels;
	int counter = 0;
	data.resize(N*COLS);
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
			for(int j = 0;j<COLS;j++)
				{
					data[counter + j*N] = row[j];
				}
			labels[counter] = (int)row[COLS];
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
	
	cudaMalloc((void**)(&d_data),N*COLS*sizeof(float));
	cudaMalloc((void**)(&d_labels),N*sizeof(int));
	cudaMalloc((void**)(&rowids),N*sizeof(int));
	
	cudaMemcpy(d_data,data.data(),N*COLS*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_labels,labels.data(),N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(rowids,h_rowids,N*sizeof(unsigned int),cudaMemcpyHostToDevice);

	ML::DecisionTree::DecisionTreeClassifier model;
	model.plant(d_data,COLS,N,d_labels,rowids,N,depth);
	model.print();

	for(int i = 0;i<N;i++)
		{
			vector<float> infer_data;
			for(int j = 0;j<COLS;j++)
				{
					infer_data.push_back(data[i + j*N]);
				}
			
			int ans = model.predict(infer_data.data());
			if(ans != labels[i])
				{
					printf("Mismatch at %d , true value %d , infered value %d\n",i,labels[i],ans);
				}
		}
	cudaFree(d_data);
	cudaFree(d_labels);
	cudaFree(rowids);
	
	return 0;
	
}
