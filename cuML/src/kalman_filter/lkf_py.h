/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace kf {
namespace linear {

size_t get_workspace_size_f32(Variables<float>& var, int _dim_x, int _dim_z, Option _solver, float *_x_est,
          float *_x_up, float *_Phi, float *_P_est, float *_P_up, float *_Q, float *_R, float *_H);
void init_f32(Variables<float>& var, int _dim_x, int _dim_z, Option _solver, float *_x_est,
          float *_x_up, float *_Phi, float *_P_est, float *_P_up, float *_Q, float *_R, float *_H,
          void* workspace, size_t& workspaceSize);
void predict_f32(Variables<float>& var);
void update_f32(Variables<float>& var, float *_z);

size_t get_workspace_size_f64(Variables<double>& var, int _dim_x, int _dim_z, Option _solver, double *_x_est,
          double *_x_up, double *_Phi, double *_P_est, double *_P_up, double *_Q, double *_R, double *_H);
void init_f64(Variables<double>& var, int _dim_x, int _dim_z, Option _solver, double *_x_est,
          double *_x_up, double *_Phi, double *_P_est, double *_P_up, double *_Q, double *_R, double *_H,
          void* workspace, size_t& workspaceSize);
void predict_f64(Variables<double>& var);
void update_f64(Variables<double>& var, double *_z);

}; // end namespace linear
}; // end namespace kf
