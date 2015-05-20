/*
 * WalkerManager
 *
 * Reference:
 * Kyrola, A. (2013). DrunkardMob: Billions of Random Walks on Just a PC.  In
 * RecSys (pp. 257–264).
 */
#ifndef WALKER_MANAGER_HPP
#define WALKER_MANAGER_HPP

#include <vector>

#include "graphchi_types.hpp"

#ifndef WSTATE_BUCKET_BIT
#define WSTATE_BUCKET_BIT 12
#endif
#define WSTATE_SOURCE_BIT (32-WSTATE_BUCKET_BIT)

namespace graphchi {
    typedef uint32_t wstate_t;

    bool wstate_compare(const wstate_t& a, const wstate_t& b) {
        return (a >> WSTATE_SOURCE_BIT) < (b >> WSTATE_SOURCE_BIT);
    }

    class WalkerManager {
    private:

        vid_t nvertices;
        vid_t source_start, source_end;
        vid_t source_len;
        size_t bucket, cursor, cur_walks;
        std::vector<std::vector<wstate_t> > walks[2];

        inline vid_t wstate_get_position(const wstate_t& state) {
            return bucket<<WSTATE_BUCKET_BIT | state >>
                WSTATE_SOURCE_BIT;
        }

        inline vid_t wstate_get_source(const wstate_t& state) {
            return (state & ((1<<WSTATE_SOURCE_BIT)-1)) + source_start;
        }

        inline wstate_t wstate_get_state(const vid_t& position, const vid_t& source) {
            return (position & ((1<<WSTATE_BUCKET_BIT)-1)) << WSTATE_SOURCE_BIT
                | (source-source_start);
        }

    public:
        WalkerManager(vid_t nvertices, vid_t source_start, vid_t source_end):
                nvertices(nvertices), source_start(source_start),
                source_end(source_end), source_len(source_end - source_start + 1),
                bucket(-1), cursor(0), cur_walks(0) {
            walks[0].resize((nvertices>>WSTATE_BUCKET_BIT) + 1);
            walks[1].resize((nvertices>>WSTATE_BUCKET_BIT) + 1);
        }

        void insert(vid_t cur_vid, vid_t source) {
            walks[1-cur_walks][cur_vid>>WSTATE_BUCKET_BIT].push_back(wstate_get_state(cur_vid,
                        source));
        }

        bool find_walkers(vid_t cur_vid, vid_t &source) {
            if (cur_vid>>WSTATE_BUCKET_BIT != bucket) {
                bucket = cur_vid>>WSTATE_BUCKET_BIT;
                std::sort(walks[cur_walks][bucket].begin(),
                        walks[cur_walks][bucket].end(), wstate_compare);
                cursor = 0;
            }

            if (walks[cur_walks][bucket].empty())
                return false;

            for (; cursor < walks[cur_walks][bucket].size(); cursor++) {
                if (wstate_get_position(walks[cur_walks][bucket][cursor]) == cur_vid) {
                    source = wstate_get_source(walks[cur_walks][bucket][cursor++]);
                    return true;
                } else if (wstate_get_position(walks[cur_walks][bucket][cursor]) > cur_vid) {
                    return false;
                }
            }

            return false;
        }

        void flip() {
            for (size_t i = 0; i < walks[cur_walks].size(); i++)
                walks[cur_walks][i].clear();
            cur_walks = 1-cur_walks;
        }
    };
}

#endif
