/**
 * @file
 * @author  Qin Liu <karlqliu@gmail.com>
 *
 * @section DESCRIPTION
 *
 * Simple personalized pagerank implementation.
 */

#include <string>
#include <fstream>
#include <cmath>

#define GRAPHCHI_DISABLE_COMPRESSION


#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

#define RANDOMRESETPROB 0.15

typedef float VertexDataType;
typedef struct {} EdgeDataType;

/**
  * Faster version of pagerank which holds vertices in memory. Used only if the number
  * of vertices is small enough.
  */
struct PagerankProgramInmem : public GraphChiProgram<VertexDataType, EdgeDataType> {

    std::vector<VertexDataType> pr[2];
    vid_t source;
    int source_outdeg;
    int cur;
    VertexDataType source_pr;

    PagerankProgramInmem(size_t nvertices, vid_t source):
            source(source), cur(0) {
        pr[0] = std::vector<VertexDataType>(nvertices, 0);
        pr[1] = std::vector<VertexDataType>(nvertices, 0);
    }

    inline void update_source(VertexDataType tmp) {
#pragma omp atomic
        source_pr += tmp;
    }

    void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
        if (ginfo.iteration > 0) {
            float sum = 0;
            for (int i = 0; i < v.num_inedges(); i++)
                sum += pr[1-cur][v.inedge(i)->vertexid];

            if (v.id() == source) {
                /* pr[cur][source] += (1 - RANDOMRESETPROB) * sum / source_outdeg; */
                update_source((1 - RANDOMRESETPROB) * sum / source_outdeg);
                if (v.outc == 0) {
                    /* pr[cur][source] += pr[1-cur][v.id()] / source_outdeg; */
                    update_source(pr[1-cur][v.id()] / source_outdeg);
                }
            } else {
                if (v.outc > 0)
                    pr[cur][v.id()] = (1 - RANDOMRESETPROB) * sum / v.outc;
                else {
                    pr[cur][v.id()] = (1 - RANDOMRESETPROB) * sum;
                    /* pr[cur][source] += pr[1-cur][v.id()] / source_outdeg; */
                    update_source(pr[1-cur][v.id()] / source_outdeg);
                }
            }
            /* pr[cur][source] += RANDOMRESETPROB * sum / source_outdeg; */
            update_source(RANDOMRESETPROB * sum / source_outdeg);
        } else if (ginfo.iteration == 0 && v.id() == source) {
            source_outdeg = v.outc;
            if (source_outdeg == 0)
                source_outdeg = 1;
            source_pr = 1.0f / source_outdeg;
        }
        if (ginfo.iteration == ginfo.num_iterations - 1) {
            /* On last iteration, multiply pr by degree and store the result */
            v.set_data(v.outc > 0 ? pr[1-cur][v.id()] * v.outc : pr[1-cur][v.id()]);
        }
    }

    void before_iteration(int iteration, graphchi_context &ginfo) {
        /* pr[cur][source] = 0; */
        source_pr = 0;
    }

    void after_iteration(int iteration, graphchi_context &ginfo) {
        pr[cur][source] = source_pr;
        cur = 1-cur;
    }
};

int main(int argc, const char ** argv) {
    graphchi_init(argc, argv);
    metrics m("ppr");
    global_logger().set_log_level(LOG_DEBUG);

    /* Parameters */
    std::string filename    = get_option_string("file"); // Base filename
    int niters              = get_option_int("niters", 10);
    bool scheduler          = false;                    // Non-dynamic version of pagerank.
    int ntop                = get_option_int("top", 20);

    /* Process input file - if not already preprocessed */
    int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));

    /* Run */
    graphchi_engine<VertexDataType, EdgeDataType> engine(filename, nshards, scheduler, m);
    engine.set_modifies_inedges(false); // Improves I/O performance.

    logstream(LOG_INFO) << "Running Pagerank by holding vertices in-memory mode!" << std::endl;
    engine.set_modifies_outedges(false);
    engine.set_disable_outedges(true);
    engine.set_only_adjacency(true);
    vid_t source_start = (vid_t) get_option_int("source_start", 0);
    vid_t source_end = (vid_t) get_option_int("source_end", 10);
    for (vid_t source = source_start; source <= source_end; source++) {
        PagerankProgramInmem program(engine.num_vertices(), source);
        engine.run(program, niters);

        /* Output top ranked vertices */
        std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
        std::cout << "Print top " << ntop << " vertices:" << std::endl;
        for(int i=0; i < (int)top.size(); i++) {
            std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
        }
    }

    metrics_report(m);
    return 0;
}

