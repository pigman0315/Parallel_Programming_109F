#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <set>
#include "../common/CycleTimer.h"
#include "../common/graph.h"
#include <vector>
#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
//#define VERBOSE
void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] == NOT_VISITED_MARKER)
            {            
                distances[outgoing] = distances[node] + 1;    
                new_frontier->vertices[__sync_fetch_and_add(&new_frontier->count,1)] = outgoing;
                
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(frontier->vertices);
    free(new_frontier->vertices);
}
int in_frontier(Graph g, int i, bool* s, vertex_set *frontier){
    int start_edge = g->incoming_starts[i];
    int end_edge = (i == g->num_nodes - 1) ? g->num_edges:g->incoming_starts[i + 1];
    for(int neighbor = start_edge; neighbor < end_edge;neighbor++){
        int incoming = g->incoming_edges[neighbor];
        if(s[incoming] == true){
            return incoming;
        }
    }
    return -1;
}
void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    int numNodes = num_nodes(g);
    int thread_num = omp_get_thread_num();
    //printf("sub routine\n");
    // create set to accelerate process of finding
    bool *s = new bool[g->num_nodes];
    #pragma omp parallel for
    for(int i = 0;i < numNodes;i++){
        s[i] = false;
    }
    //#pragma omp parallel for
    for(int j = 0;j < frontier->count;j++){
        s[frontier->vertices[j]] = true;
    }
    bool flg = false;

    #pragma omp parallel for
    for(int i = 0;i < numNodes;i++){
        if(distances[i] == NOT_VISITED_MARKER){ 
            // find if incoming node is in frontier   
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1) ? g->num_edges:g->incoming_starts[i + 1];
            for(int neighbor = start_edge; neighbor < end_edge;neighbor++){
                int incoming = g->incoming_edges[neighbor];
                if(s[incoming] == true){
                    distances[i] = distances[incoming]+1;
                    new_frontier->vertices[__sync_fetch_and_add(&new_frontier->count,1)] = i;
                    break;
                }
            }
        }
    }
    free(s);
}
void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(frontier->vertices);
    free(new_frontier->vertices);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        //if(frontier->count >= graph->num_nodes / 2)
        if(frontier->count >= graph->num_nodes / 10)
            bottom_up_step(graph, frontier, new_frontier, sol->distances);
        else
            top_down_step(graph, frontier, new_frontier, sol->distances);
#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    free(frontier->vertices);
    free(new_frontier->vertices);
}