import time
import igraph
import numpy


def main(args):
    startTime = time.time()

    size = int(args.get('size', '10000'))
    graph = igraph.Graph.Barabasi(size, 10)
  
    process_begin = time.time()
    result = graph.bfs(0)
    process_end = time.time()

    process_time = process_end - process_begin

    return {
            'startTime': startTime,
            'latency': process_time,
            'functionTime': process_end - startTime
    }
