#pragma once

void init_hmm_f32(HMM<float> &hmm);

void forward_f32(HMM<float>& hmm,
                 float* dX,
                 int* len_array,
                 int nObs);

void backward_f32(HMM<float>& hmm,
                  float* dX,
                  int* len_array,
                  int nObs);

void setup_hmm_f32(HMM<float> &hmm);

void free_hmm_f32(HMM<float> &hmm);
