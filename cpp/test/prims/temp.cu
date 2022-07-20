// #include <cuml/metrics/metrics.hpp>
#include <gtest/gtest.h>
#include "test_utils.h"
#include <metrics/davies_bouldin_score.cuh>

namespace MLCommon {
namespace Metrics {

TEST(daviesBouldinScore, functionalityCheck2Classes) 
{ 
  auto handle = raft::handle_t{};
  int nRows = 5;
  int nCols = 2;
  int nLabels = 2;
  int h_labels[nRows] = {0,0,0,1,1};
  float h_Xin[nRows][nCols] = {{1,1}, {1,2}, {1,3}, {1,4}, {1,5}};
  auto stream = handle.get_stream();

  rmm::device_uvector<float> d_Xin(nRows * nCols, stream);
  raft::update_device(d_Xin.data(), &h_Xin[0][0], (int)nRows*nCols, stream);

  rmm::device_uvector<int> d_labels(nRows, stream);
  raft::update_device(d_labels.data(), &h_labels[0], nRows, stream);

  float dbscore = davies_bouldin_score(handle,
                                       d_Xin.data(),
                                       nRows,
                                       nCols,
                                       d_labels.data(),
                                       nLabels,
                                       stream,
                                       raft::distance::DistanceType::L2Unexpanded);

  float sklearn_dbscore = 0.46666;
  ASSERT_NEAR(dbscore, sklearn_dbscore, 0.0001);

}

TEST(daviesBouldinScore, functionalityCheck3classes) 
{ 
  auto handle = raft::handle_t{};
  int nRows = 6;
  int nCols = 2;
  int nLabels = 3;
  int h_labels[nRows] = {0,0,1,1,2,2};
  float h_Xin[nRows][nCols] = {{1,1}, {1,2}, {1,3}, {1,4}, {1,5}, {1,6}};
  auto stream = handle.get_stream();

  rmm::device_uvector<float> d_Xin(nRows * nCols, stream);
  raft::update_device(d_Xin.data(), &h_Xin[0][0], (int)nRows*nCols, stream);

  rmm::device_uvector<int> d_labels(nRows, stream);
  raft::update_device(d_labels.data(), h_labels, nRows, stream);

  float dbscore = davies_bouldin_score(handle,
                                       d_Xin.data(),
                                       nRows,
                                       nCols,
                                       d_labels.data(),
                                       nLabels,
                                       stream,
                                       raft::distance::DistanceType::L2Unexpanded);

  float sklearn_dbscore = 0.5;
  ASSERT_NEAR(dbscore, sklearn_dbscore, 0.0001);
  
}

TEST(daviesBouldinScore, functionality1pointincluster) 
{ 
  auto handle = raft::handle_t{};
  int nRows = 5;
  int nCols = 2;
  int nLabels = 3;
  int h_labels[nRows] = {0,0,1,1,2};
  float h_Xin[nRows][nCols] = {{1,1}, {1,2}, {1,3}, {1,4}, {1,5}};

  auto stream = handle.get_stream();

  rmm::device_uvector<float> d_Xin(nRows * nCols, stream);
  raft::update_device(d_Xin.data(), &h_Xin[0][0], (int)nRows*nCols, stream);

  rmm::device_uvector<int> d_labels(nRows, stream);
  raft::update_device(d_labels.data(), h_labels, nRows, stream);

  float dbscore = davies_bouldin_score(handle,
                                       d_Xin.data(),
                                       nRows,
                                       nCols,
                                       d_labels.data(),
                                       nLabels,
                                       stream,
                                       raft::distance::DistanceType::L2Unexpanded);

  float sklearn_dbscore = 0.44444;
  ASSERT_NEAR(dbscore, sklearn_dbscore, 0.0001);
  
}

TEST(daviesBouldinScore, functionality3cols) 
{ 
  auto handle = raft::handle_t{};
  int nRows = 5;
  int nCols = 3;
  int nLabels = 3;
  int h_labels[nRows] = {0,0,1,1,2};
  float h_Xin[nRows][nCols] = {{1,1,1}, {1,2,2}, {1,3,3}, {1,4,1}, {1,5,2}};

  auto stream = handle.get_stream();

  rmm::device_uvector<float> d_Xin(nRows * nCols, stream);
  raft::update_device(d_Xin.data(), &h_Xin[0][0], (int)nRows*nCols, stream);

  rmm::device_uvector<int> d_labels(nRows, stream);
  raft::update_device(d_labels.data(), h_labels, nRows, stream);

  float dbscore = davies_bouldin_score(handle,
                                       d_Xin.data(),
                                       nRows,
                                       nCols,
                                       d_labels.data(),
                                       nLabels,
                                       stream,
                                       raft::distance::DistanceType::L2Unexpanded);

  float sklearn_dbscore = 0.838667;
  ASSERT_NEAR(dbscore, sklearn_dbscore, 0.0001);
  
}

};
};
