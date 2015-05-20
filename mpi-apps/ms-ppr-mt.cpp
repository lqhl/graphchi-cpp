/*
 * Multi-Source Personalized PageRank with multi-thread support using a
 * WalkManager similar to DrunkardMob
 */
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include <mpi.h>

/* Uncomment this to use sparse_hash_map */
// #define SPARSE_TABLE

#ifndef SPARSE_TABLE
#include <sparsehash/dense_hash_map>
using google::dense_hash_map;
#else
#include <sparsehash/sparse_hash_map>
using google::sparse_hash_map;
#endif

#define REVERSE_GRAPH

#include "graphchi_basic_includes.hpp"
#include "api/walker_manager.hpp"

using namespace graphchi;

typedef int VertexDataType;
typedef struct {} EdgeDataType;

bool compare(const std::pair<int, float>& firstElem, const std::pair<int, float>& secondElem) {
      return firstElem.second > secondElem.second;
}

bool scheduler = false;

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
 * class. The main logic is usually in the update function.
 */
class PageRankProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {
private:
    metrics &m;
    vid_t source_start, source_end;
    vid_t source_len;
    int R;
    int niters;
    unsigned int seed;
    WalkerManager walker_manager;
#ifndef SPARSE_TABLE
    std::vector <dense_hash_map <vid_t, uint16_t> > counter;
#else
    std::vector <sparse_hash_map <vid_t, uint16_t> > counter;
#endif
    std::vector <int> total;

public:
    PageRankProgram(metrics &_m, vid_t nvertices, vid_t source_start, vid_t
            source_end, int R, int niters):
            m(_m), source_start(source_start), source_end(source_end),
            source_len(source_end - source_start + 1), R(R),
            niters(niters), walker_manager(nvertices, source_start, source_end),
            counter(source_len), total(source_len) {
        assert(source_start <= source_end);

#ifndef SPARSE_TABLE
        for (size_t i = 0; i < counter.size(); i++)
            counter[i].set_empty_key(-1);
#endif

        for (vid_t source = source_start; source <= source_end; source++)
            for (int j = 0; j < R; j++) {
                walker_manager.insert(source, source);
            }
        walker_manager.flip();

        seed = source_start;
    }

    /**
     *  Vertex update function.
     */
    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &gcontext) {
        vid_t source = 0;
        while (walker_manager.find_walkers(v.id(), source)) {
            counter[source-source_start][v.id()]++;
            total[source-source_start]++;
            if ((float)rand_r(&seed) / (float)RAND_MAX <= 0.15) {
                // do nothing
            } else if (v.num_inedges() > 0) {
                vid_t next = v.inedge(rand_r(&seed) % v.num_inedges())->vertex_id();
                walker_manager.insert(next, source);
                if (scheduler)
                    gcontext.scheduler->add_task(next);
            } else {
                walker_manager.insert(source, source);
                if (scheduler)
                    gcontext.scheduler->add_task(source);
            }
        }
    }

    /**
     * Called before an iteration starts. Not implemented.
     */
    void before_iteration(int iteration, graphchi_context &ginfo) {
    }

    /**
     * Called after an iteration has finished. Not implemented.  */
    void after_iteration(int iteration, graphchi_context &ginfo) {
        walker_manager.flip();
    }

    /**
     * Called before an execution interval is started. Not implemented.
     */
    void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {
    }

    void after_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {
    }

    std::vector<std::pair<vid_t, float> > result(vid_t source) {
        source -= source_start;
        std::vector<std::pair<vid_t, float> > ret;
#ifndef SPARSE_TABLE
        for (dense_hash_map <vid_t, uint16_t>::iterator it = counter[source].begin(); it != counter[source].end(); it++) {
#else
        for (sparse_hash_map <vid_t, uint16_t>::iterator it = counter[source].begin(); it != counter[source].end(); it++) {
#endif
            ret.push_back(std::make_pair(it->first, it->second / float(total[source])));
        }
        printf("total steps: %d\n", total[source]);
        return ret;
    }
};

int main(int argc, char ** argv) {
    /* GraphChi initialization will read the command line
       arguments and the configuration file. */
    graphchi_init(argc, (const char**)argv);

    /* Metrics object for keeping track of performance counters
       and other information. Currently required. */
    metrics m("ms-ppr-drunkardmob");

    /* Basic arguments for application */
    std::string filename = get_option_string("file");  // Base filename
    int niters           = get_option_int("niters", 10); // Number of iterations (max)
    scheduler            = get_option_int("scheduler", 0);
    bool print           = get_option_int("print", 0);

    /* Process input file - if not already preprocessed */
    int nshards          = convert_if_notexists<EdgeDataType>(filename,
            get_option_string("nshards", "auto"));

    int num_tasks, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    logstream(LOG_INFO) << "MPI rank: " << rank << ", num tasks: " << num_tasks
        << std::endl;

    /* Run */
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
    engine.set_disable_vertexdata_storage();
    engine.set_only_adjacency(true);
    engine.set_disable_outedges(true);

    vid_t source_start = (vid_t) get_option_int("source_start", 0);
    vid_t source_end = (vid_t) get_option_int("source_end", 999);
    vid_t source_range = source_end - source_start + 1;
    vid_t sources_per_node = source_range / num_tasks + ((source_range % num_tasks == 0) ? 0 : 1);

    int R = get_option_int("R", 1000);

    vid_t cur_start = source_start + rank * sources_per_node;
    vid_t cur_end = std::min(cur_start + sources_per_node - 1, source_end);
    logstream(LOG_INFO) << "source range: " << cur_start << " -- " << cur_end
        << std::endl;

    /* int num_programs = get_option_int("num_programs", 1); */
    int num_programs = get_option_int("execthreads", omp_get_max_threads());
    source_range = cur_end - cur_start + 1;
    vid_t sources_per_thread = source_range / num_programs + ((source_range % num_programs == 0) ? 0 : 1);

    std::vector< graphchi_engine<VertexDataType, EdgeDataType>::prog_t*> userprogramPool;
    for (int i = 0; i < num_programs; i++) {
        vid_t start = cur_start + i * sources_per_thread;
        vid_t end = std::min(start + sources_per_thread - 1, cur_end);
        logstream(LOG_INFO) << "thread " << i << ", source range: " << start <<
            " -- " << end << std::endl;
        userprogramPool.push_back(new PageRankProgram(m, engine.num_vertices(), start, end, R, niters));
    }
    engine.run(userprogramPool, niters);

    for (int i = 0; i < num_programs; i++) {
        if (print) {
            vid_t start = cur_start + i * sources_per_thread;
            vid_t end = std::min(start + sources_per_thread - 1, cur_end);
            for (vid_t source = start; source <= std::min(end, start + 9u); source++) {
                std::vector<std::pair<vid_t, float> > ret = ((PageRankProgram *) userprogramPool[i])->result(source);
                std::sort(ret.begin(), ret.end(), compare);
                for (int i = 0; i < std::min((int) ret.size(), 20); i++)
                    printf("\t%d", ret[i].first);
                printf("\n");
                for (int i = 0; i < std::min((int) ret.size(), 20); i++)
                    printf("\t%.4f", ret[i].second);
                printf("\n");
            }
        }
        delete userprogramPool[i];
    }

    /* Report execution metrics */
    metrics_report(m);
    MPI_Finalize();
    return 0;
}

