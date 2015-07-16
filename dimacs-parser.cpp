#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " IN OUT" << endl;
    exit(1);
  }

  vector<int> nodes, edges;
  
  ifstream in(argv[1]);
  assert(in.is_open());
  string buf;
  getline(in, buf);
  int n, m;
  sscanf(buf.c_str(), "%d %d", &n, &m);
  nodes.reserve(n + 1);
  edges.reserve(m);
  nodes.push_back(0);
  for (int i = 0; i < n; ++i) {
    getline(in, buf);
    istringstream parser(buf);
    int neighbor;
    while (parser >> neighbor) {
      edges.push_back(neighbor - 1);
    }
    nodes.push_back(edges.size());
  }
  assert(edges.size() == m);

  ofstream out(argv[2], ios::binary);    
  assert(out.is_open());
  out.write((char*)&n, sizeof(int));
  out.write((char*)&m, sizeof(int));
  out.write((char*)nodes.data(), nodes.size() * sizeof(int));
  out.write((char*)edges.data(), edges.size() * sizeof(int));
}
