#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <string.h>
#include <omp.h>
#include <utility>
#include <vector>
#include "../common/CycleTimer.h"
#include "../common/graph.h"
using namespace std;

void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / (double)numNodes;
  bool converged = false;
  int *no_outedge_node_ary = new int[numNodes];
  int no_outedge_node_cnt = 0;
  double *score_new, *score_old, *zero_ary;
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }
  // 
  zero_ary = new double[numNodes];
  score_new = new double[numNodes];
  score_old = new double[numNodes];
  // initialization
  #pragma omp parallel for
  for(int i = 0;i < numNodes;i++){
    score_old[i] = equal_prob;
    score_new[i] = 0.0;
    zero_ary[i] = 0.0;
    no_outedge_node_ary[i] = -1;
  }
  // find which nodes have no out-edge
  #pragma omp parallel for
  for(int i = 0;i < numNodes;i++){
    int s = g->outgoing_starts[i];
    int e = ((i == numNodes - 1)? g->num_edges:g->outgoing_starts[i+1]);
    if(e-s == 0){
      #pragma omp critical
      {
       no_outedge_node_ary[no_outedge_node_cnt++] = i;
      }
    }
  }
  
  //printf("no: %d\n",no_outedge_node_cnt);
  while(!converged){
    //compute how much per-node scores have changed 
    double global_diff = 0.0;
    //sum over all nodes v in graph with no outgoing edges
    double sum_no_out = 0.0;
    #pragma omp parallel for reduction(+:sum_no_out)
    for(int j = 0;j < no_outedge_node_cnt;j++){
      sum_no_out += damping * score_old[no_outedge_node_ary[j]] / (double)numNodes;
    }
    // compute score_new[vi] for all nodes vi:
    #pragma omp parallel for reduction(+:global_diff)
    for(int i = 0;i < numNodes;i++){
      // sum over all nodes vj reachable from incoming edges
      int start_edge = g->incoming_starts[i];
      int end_edge;
      if(i == numNodes - 1){
        end_edge = g->num_edges;
      }
      else{
        end_edge = g->incoming_starts[i+1];
      }
      // attempt to reach all neighbors of vi
      //sum over all nodes vj reachable from incoming edges
      for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
      {
          int outgoing = g->incoming_edges[neighbor];
          int s = g->outgoing_starts[outgoing];
          int e;
          if(outgoing == numNodes - 1){
            e = g->num_edges;
          }
          else{
            e = g->outgoing_starts[outgoing+1];
          }
          score_new[i] += score_old[outgoing] / (double)abs(e-s);
      }
      score_new[i] = (double)(damping * score_new[i]) + (double)(1.0-damping)/(double)numNodes;
      score_new[i] += sum_no_out;
      //sum over all nodes v in graph with no outgoing edges
      // for(int j = 0;j < no_outedge_node_cnt;j++){
      //   score_new[i] += damping * score_old[no_outedge_node_ary[j]] / (double)numNodes;
      // }
      global_diff += abs(score_new[i] - score_old[i]);
    }
    //quit once algorithm has converged
    converged = (global_diff < convergence);
    // swap & initialization
    memcpy(score_old,score_new,sizeof(double)*numNodes);
    memcpy(score_new,zero_ary,sizeof(double)*numNodes);
  }//end while
  memcpy(solution,score_old,sizeof(double)*numNodes);
  free(score_old);
  free(score_new);
  free(zero_ary);
  free(no_outedge_node_ary);
}
