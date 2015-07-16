# CUDA BFS
This repository contains a simple, *O(n+m)* work, BFS implementation for CUDA, using the Modern GPU library.

Minimal working example:

```
make
wget http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn20.graph.bz2
bzip -d kron_g500-logn20.graph.bz2
./dimacs-parser kron_g500-logn20.graph kron_g500-logn20.graph.bin
./bfs-mgpu kron_g500-logn20.graph.bin
```
