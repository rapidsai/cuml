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

template<typename T>
struct Question
{
  int column;
  T value;
};

struct TreeNode
{
  TreeNode *left = NULL;
  TreeNode *right = NULL;
  bool leaf = false;
  Question<float> question;
};

class DecisionTreeClassifier
{
public:
  TreeNode *root;
  void fit(float *data,int ncols,int nrows,float colper,int *lables)
  {
    root = grow_tree(data,ncols,nrows,colper,lables);
  }

  void plant(float *data,int ncols,int nrows,float colper,int *lables)
  {
    root = grow_tree(data,ncols,nrows,colper,lables);
  }

  TreeNode* grow_tree(float *data,int ncols,int nrows,float colper,int *lables)
  {
    TreeNode *node = new TreeNode();
    return node;
  }
  
};
