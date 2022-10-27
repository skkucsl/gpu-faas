import time
import igraph
import os

def main(args):
    startTime = time.time()
    size = int(args.get('size', '10000'))
    
    graph_generating_begin = time.time()
    graph = igraph.Graph.Barabasi(size, 10)
    graph_generating_end = time.time()

    process_begin = time.time()
    result = graph.bfs(0)
    process_end = time.time()

    graph_generating_time = graph_generating_end - graph_generating_begin
    process_time = process_end - process_begin

    return {
            'startTime': startTime,
            'latency': process_time,
            'functionTime': process_end-startTime
    }
