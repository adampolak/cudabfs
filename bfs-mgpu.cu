#include <moderngpu.cuh>
using namespace mgpu;

#include <queue>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cstdio>
using namespace std;

__global__ void UpdateDistanceAndVisitedKernel(
    const int* __restrict__ frontier, int frontier_size, int d,
    int* distance, int* visited) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < frontier_size; i += step) {
    distance[frontier[i]] = d;
    atomicOr(visited + (frontier[i] >> 5), 1 << (frontier[i] & 31));
  }
}

__global__ void CalculateFrontierStartsAndDegreesKernel(
    const int* __restrict__ nodes, const int* __restrict__ frontier, int n,
    int* node_frontier_starts, int* node_frontier_degrees) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < n; i += step) {
    node_frontier_starts[i] = nodes[frontier[i]];
    node_frontier_degrees[i] = nodes[frontier[i] + 1] - nodes[frontier[i]];
  }
}

__global__ void AdvanceFrontierPhase1Kernel(
      const int* __restrict__ edge_frontier, int edge_frontier_size,
      const int* __restrict__ visited,
      int* parent, int* edge_frontier_success) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < edge_frontier_size; i += step) {
    int v = edge_frontier[i];
    int success = (((visited[v >> 5] >> (v & 31)) & 1) == 0 && parent[v] == -1) ? 1 : 0;
    if (success)
      parent[edge_frontier[i]] = i;
    edge_frontier_success[i] = success;
  }
}

__global__ void AdvanceFrontierPhase2Kernel(
      const int* __restrict__ edge_frontier, int edge_frontier_size,
      const int* __restrict__ parent, int* edge_frontier_success) {
  int from = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  for (int i = from; i < edge_frontier_size; i += step)
    if (edge_frontier_success[i] && parent[edge_frontier[i]] != i)
      edge_frontier_success[i] = 0;
}

void ParallelBFS(
    int n, int m, MGPU_MEM(int) nodes, MGPU_MEM(int) edges, int source,
    MGPU_MEM(int) distance, CudaContext& context) {
  MGPU_MEM(int) visited = context.Fill((n + 31) / 32, 0);
  MGPU_MEM(int) parent = context.Fill(n, -1);
  MGPU_MEM(int) node_frontier = context.Malloc<int>(n);
  MGPU_MEM(int) node_frontier_starts = context.Malloc<int>(n);  
  MGPU_MEM(int) node_frontier_degrees = context.Malloc<int>(n);
  MGPU_MEM(int) edge_frontier = context.Malloc<int>(m);
  MGPU_MEM(int) edge_frontier_success = context.Malloc<int>(m);
  node_frontier->FromHost(&source, 1);
  int node_frontier_size = 1;
  for (int d = 0; node_frontier_size > 0; ++d) {
    // cerr << "d = " << d << " frontier_size = " << node_frontier_size << endl;
    // PrintArray(*node_frontier, "%d", 10);
    UpdateDistanceAndVisitedKernel<<<128, 128, 0, context.Stream()>>>(
        node_frontier->get(), node_frontier_size, d,
        distance->get(), visited->get());
    CalculateFrontierStartsAndDegreesKernel<<<128, 128, 0, context.Stream()>>>(
        nodes->get(), node_frontier->get(), node_frontier_size,
        node_frontier_starts->get(), node_frontier_degrees->get());
    int edge_frontier_size;
    ScanExc(
        node_frontier_degrees->get(), node_frontier_size,
        &edge_frontier_size, context);
    IntervalGather(
        edge_frontier_size, node_frontier_starts->get(),
        node_frontier_degrees->get(), node_frontier_size, edges->get(),
        edge_frontier->get(), context);
    AdvanceFrontierPhase1Kernel<<<128, 128, 0, context.Stream()>>>(
        edge_frontier->get(), edge_frontier_size, visited->get(),
        parent->get(), edge_frontier_success->get());
    AdvanceFrontierPhase2Kernel<<<128, 128, 0, context.Stream()>>>(
        edge_frontier->get(), edge_frontier_size,
        parent->get(), edge_frontier_success->get());
    ScanExc(
        edge_frontier_success->get(), edge_frontier_size,
        &node_frontier_size, context);
    IntervalExpand(
        node_frontier_size, edge_frontier_success->get(),
        edge_frontier->get(), edge_frontier_size,
        node_frontier->get(), context);
  }
}

typedef unsigned long long uint64_t;

uint64_t CalculateChecksum(const vector<int>& distance) {
  uint64_t checksum = 0;
  for (int i = 0; i < distance.size(); ++i)
    if (distance[i] != -1)
      checksum += (uint64_t)i * (uint64_t)distance[i];
  return checksum;
}

uint64_t Time() {
  timespec tp;
  clock_gettime(CLOCK_MONOTONIC_RAW, &tp);
  return (tp.tv_nsec + (uint64_t)1000000000 * tp.tv_sec) / 1000000;
}

uint64_t ParallelBFS(
    const vector<int>& nodes, const vector<int>& edges, int source) {
  ContextPtr context = CreateCudaDevice(0);
  MGPU_MEM(int) dev_nodes = context->Malloc(nodes);
  MGPU_MEM(int) dev_edges = context->Malloc(edges);
  MGPU_MEM(int) dev_distance = context->Fill(nodes.size() - 1, -1);
  uint64_t t = Time();
  ParallelBFS(
      nodes.size() - 1, edges.size(), dev_nodes, dev_edges, source,
      dev_distance, *context);
  t = Time() - t;
  cerr << "GPU: " << t << " ms" << endl;
  vector<int> distance;
  dev_distance->ToHost(distance, nodes.size() - 1);
  return CalculateChecksum(distance);
}

uint64_t SequentialBFS(
    const vector<int>& nodes, const vector<int>& edges, int source) {
  vector<int> distance(nodes.size() - 1, -1);
  uint64_t t = Time();
  distance[source] = 0;
  queue<int> q;
  q.push(source);
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int i = nodes[u]; i < nodes[u + 1]; ++i) {
      int v = edges[i];
      if (distance[v] == -1) {
        distance[v] = distance[u] + 1;
        q.push(v);
      }
    }
  }
  t = Time() - t;
  cerr << "CPU: " << t << " ms" << endl;
  return CalculateChecksum(distance);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " GRAPH" << endl;
    exit(1);
  }

  ifstream in(argv[1], ios::binary);  
  assert(in.is_open());
  int n, m;
  in.read((char*)&n, sizeof(int));
  in.read((char*)&m, sizeof(int));
  vector<int> nodes(n + 1), edges(m);
  in.read((char*)nodes.data(), nodes.size() * sizeof(int));
  in.read((char*)edges.data(), edges.size() * sizeof(int));

  for (int i = 0; i < 5; ++i) {
    int source = rand() % n;
    uint64_t seqsum = SequentialBFS(nodes, edges, source);
    uint64_t parsum = ParallelBFS(nodes, edges, source);
    assert(seqsum == parsum);
  }
}
