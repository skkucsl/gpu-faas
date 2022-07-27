import time
import igraph

def init_time(args):
    startTime = time.time()

    size = int(args.get('size', '100'))

    graph_generating_begin = time.time()
    graph = igraph.Graph.Barabasi(size, 10)
    graph_generating_end = time.time()

    process_begin = time.time()
    result = graph.pagerank()
    process_end = time.time()

    graph_generating_time = graph_generating_end - graph_generating_begin
    process_time = process_end - process_begin

    return {
            'result': result[0],
            'measurement': {
                'graph_generating_time': graph_generating_time,
                'compute_time': process_time
            },
            'startTime': startTime
    }
