#include <vlfd_experiments/KDTreeVectorOfVectorsAdaptor.h>
#include <vlfd_experiments/helper.h>
#include <boost/python.hpp>
#include <nanoflann.hpp>
#include <algorithm>
#include <sstream>
#include <random>
#include <chrono>
#include <ctime>
#include <cmath>

namespace bp = boost::python;
using namespace nanoflann;
using namespace std::chrono;

high_resolution_clock::time_point t1, t2;
duration<double, std::milli> dur;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

typedef KDTreeVectorOfVectorsAdaptor<VStateVec, double>  KDTreeT;
typedef std::pair<double, double> Pair;
typedef std::vector<double> Pose;
typedef std::vector<Pose> Path;
typedef std::vector<Path> PathVec;
typedef std::vector<size_t> PatternElement;
typedef std::vector<PatternElement> Pattern;
typedef std::vector<Pattern> MultiPattern;

typedef float ScoreT;

class Match {
public:
  ScoreT score;
  int idx;
  int idy;
  int idz;
  std::vector<size_t> path_ids;
};
typedef std::vector<Match> Matches;

class Placement {
public:
  Pose getBase() { return base; }
  Pose base;

  PathVec getPaths() { return paths; }
  PathVec paths;

  ScoreT getScore() { return score; }
  ScoreT score;
};
typedef std::vector<Placement> Placements;


class ReachabilityDB {
public:
  ReachabilityDB(const std::string& model_name, const std::string& filename)
    : model_name_(model_name)
  {
    if(model_name_ == "baxter_arm") {
      lower_ = {-1.700, -2.146, -3.053, -0.04, -3.058, -1.569, -3.058};
      upper_ = { 1.700,  1.046,  3.053, 2.617,  3.058,  2.093,  3.058};
    }
    else {
      std::cout << "UNKNOWN MODEL!" << std::endl;
      exit(-1);
    }
    for(size_t i = 0; i < lower_.size(); ++i) {
      midpt_.push_back(0.5 * (upper_[i] + lower_[i]));
      halfr_.push_back(0.5 * (upper_[i] - lower_[i]));
    }
    if(!load(filename)) {
      std::cout << "RDB DOES NOT EXIST!" << std::endl;
      exit(-1);
    }
    stat();
  }
  ~ReachabilityDB() {}

public:
  bool load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if(ifs.is_open()) {
      std::cout << "LOADING REACHABILITY DATABASE.." << std::endl;
      t1 = high_resolution_clock::now();
      bounds_.push_back(Pair());
      bounds_.push_back(Pair());
      bounds_.push_back(Pair());
      for(auto itb = bounds_.begin(); itb != bounds_.end(); ++itb) {
        ifs.read(reinterpret_cast<char*>(&(itb->first)), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&(itb->second)), sizeof(double));
      }
      ifs.read(reinterpret_cast<char*>(&step_size_), sizeof(double));
      ifs.read(reinterpret_cast<char*>(&num_x_), sizeof(size_t));
      ifs.read(reinterpret_cast<char*>(&num_y_), sizeof(size_t));
      ifs.read(reinterpret_cast<char*>(&num_z_), sizeof(size_t));
      ifs.read(reinterpret_cast<char*>(&num_d_), sizeof(size_t));
      ifs.read(reinterpret_cast<char*>(&num_r_), sizeof(size_t));
      ifs.read(reinterpret_cast<char*>(&num_f_), sizeof(size_t));
      num_p_ = num_x_ * num_y_ * num_z_;
      num_o_ = num_d_ * num_r_;

      double v;
      quats_.resize(num_o_);
      for(auto itq = quats_.begin(); itq != quats_.end(); ++itq) {
        itq->resize(4);
        for(auto itqe = itq->begin(); itqe != itq->end(); ++itqe) {
          ifs.read(reinterpret_cast<char*>(&v), sizeof(double));
          *itqe = v;
        }
        oris_.push_back(quatMat(*itq));
      }

      ScoreT s;
      size_t hmm = 0;
      elems_.resize(num_o_);
      for(auto it1 = elems_.begin(); it1 != elems_.end(); ++it1) {
        it1->resize(num_p_);
        for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
          ifs.read(reinterpret_cast<char*>(&s), sizeof(ScoreT));
          *it2 = s;
          if(s > 0) hmm++;
        }
      }
      ifs.close();
      quats_index_.reset(new KDTreeT(4, quats_, 10));
      dur = high_resolution_clock::now() - t1;
      std::cout << "TIME (ms): " << dur.count() << std::endl;
      std::cout << "Loaded RDB built with " << num_f_ << " FKs." << std::endl;
      std::cout << "Reachable: " << hmm << " / " << num_p_ * num_o_ << std::endl;
      return true;
    }
    return false;
  };
  void stat() {
    std::vector<ScoreT> all;
    all.reserve(elems_.size() * num_x_ * num_y_ * num_z_);
    for(auto it1 = elems_.begin(); it1 != elems_.end(); ++it1) {
      all.insert(all.end(), it1->begin(), it1->end());
    }
    std::cout << "ALL: " << all.size() << std::endl;
    std::cout << "MAX: " << *(std::max_element(all.begin(), all.end())) << std::endl;
    std::cout << "MIN: " << *(std::min_element(all.begin(), all.end())) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e > 0; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e > 0 && e < 0.01; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e >= 0.01 && e < 0.05; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e >= 0.05 && e < 0.1; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e >= 0.1 && e < 0.15; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e >= 0.15 && e < 0.2; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e >= 0.20 && e < 0.25; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                       [](ScoreT e){ return e >= 0.25; }) << std::endl;
  }
  size_t getClosestOriIdx(const arma::mat& ori) {
    return getClosestOriIdx(quatFromMat(ori));
  }
  size_t getClosestOriIdx(const std::vector<double>& quat) {
    std::vector<double> qt = quat;
    if(qt[3] < 0)
      qt = qt * (-1);
    const size_t num_results = 1;
    std::vector<size_t>   ret_indexes(num_results);
    std::vector<double> out_dists_sqr(num_results);
    quats_index_->query(&qt[0], num_results, &ret_indexes[0], &out_dists_sqr[0]);
    return ret_indexes[0];
  }
  size_t getNumOris() {
    return oris_.size();
  }
  size_t getXIdx(double v) {
      return size_t(0.5 + (v - bounds_[0].first) / step_size_);
  }
  size_t getYIdx(double v) {
      return size_t(0.5 + (v - bounds_[1].first) / step_size_);
  }
  size_t getZIdx(double v) {
      return size_t(0.5 + (v - bounds_[2].first) / step_size_);
  }
  size_t getPosIdx(size_t idx, size_t idy, size_t idz) {
    return idz + (num_z_ * (idy + num_y_ * idx));
  }
  ScoreT getReachabilityElement(size_t ido, int idx, int idy, int idz) {
    if(ido >= num_o_ || idx >= num_x_ || idy >= num_y_ || idz >= num_z_) return 0;
    else return elems_[ido][getPosIdx(idx, idy, idz)];
  }
  Pattern getPattern(const Path& path,
                     const Pair& zbound,
                     double rotz) {
    Path npath;
    arma::mat mrotz = rotZMat(rotz);
    for(auto it = path.begin(); it != path.end(); ++it) {
      npath.push_back(transformFromMat(mrotz * transformMat(*it)));
    }
    return getPattern(npath, zbound);
  }
  Pattern getPattern(const Path& path,
                     const Pair& zbound) {
    Pattern pattern;
    double xmin, ymin, zmin;
    xmin = ymin = 1e9;
    zmin = bounds_[2].first + zbound.first;
    for(auto it = path.begin(); it != path.end(); ++it) {
      if((*it)[0] < xmin) xmin = (*it)[0];
      if((*it)[1] < ymin) ymin = (*it)[1];
    }
    // Make the first point match exactly
    double xoffset = step_size_ * (fmod(0.5 + (path[0][0] - xmin) / step_size_, 1.0) - 1.5);
    double yoffset = step_size_ * (fmod(0.5 + (path[0][1] - ymin) / step_size_, 1.0) - 1.5);
    xmin += xoffset;
    ymin += yoffset;
    for(auto it = path.begin(); it != path.end(); ++it) {
      PatternElement pi;
      pi.push_back(getClosestOriIdx(std::vector<double>(it->begin()+3, it->end())));
      pi.push_back(size_t(0.5 + ((*it)[0] - xmin) / step_size_));
      pi.push_back(size_t(0.5 + ((*it)[1] - ymin) / step_size_));
      pi.push_back(size_t(0.5 + ((*it)[2] - zmin) / step_size_));
      pattern.push_back(pi);
    }
    return pattern;
  }
  ScoreT scorePattern(const Pattern& p, int idx, int idy, int idz) {
    ScoreT score = 1e9;
    for(auto it = p.begin(); it != p.end(); ++it) {
      ScoreT rch = getReachabilityElement((*it)[0], (*it)[1]+idx, (*it)[2]+idy, (*it)[3]+idz);
      if(score > rch) score = rch;
      if(score == 0) break;
    }
    return score;
  }
  Matches searchPattern(const Pattern& pattern) {
    int idxmax = 0;
    int idymax = 0;
    for(auto it = pattern.begin(); it != pattern.end(); ++it) {
      if((*it)[1] > idxmax) idxmax = (*it)[1];
      if((*it)[2] > idymax) idymax = (*it)[2];
    }
    Matches matches;
    for(int idx = -idxmax; idx < num_x_; ++idx) {
      for(int idy = -idymax; idy < num_y_; ++idy) {
        Match match;
        match.score = scorePattern(pattern, idx, idy, 0);
        match.idx = idx;
        match.idy = idy;
        match.path_ids.push_back(0);
        if(match.score > 0) {
          matches.push_back(match);
        }
      }
    }
    std::sort(matches.begin(), matches.end(), 
              [](const Match& a, const Match& b) { return a.score > b.score; });
    //return Matches(matches.begin(), matches.begin() + std::min(size_t(10), matches.size()));
    return matches;
  }
  Placements getPathPlacements(const Path& path,
                               const Pair& zbound,
                               const Pair& rbound,
                               const arma::mat& extension) {
    arma::mat ext_inv = arma::inv(extension);
    Path npath;
    for(auto it = path.begin(); it != path.end(); ++it) {
      npath.push_back(transformFromMat(transformMat(*it) * ext_inv));
    }
    return getPathPlacements(npath, zbound, rbound);
  }
  Placements getPathPlacements(const Path& path,
                               const Pair& zbound,
                               const Pair& rbound) {
    assert(zbound.first == zbound.second); // For now, only support fixed z value
    assert(rbound.first <  rbound.second); // Require a valid base z rot range, assuming upright
    Placements placements;
    std::vector<double> rotzs = linspace<double>(rbound.first, rbound.second, 10, false);
    for(auto it = rotzs.begin(); it != rotzs.end(); ++it) {
      Pattern pattern = getPattern(path, zbound, *it);
      Matches matches = searchPattern(pattern);
      for(auto itm = matches.begin(); itm != matches.end(); ++itm) {
        std::vector<double> wp = std::vector<double>(path[0].begin(), path[0].begin()+3);
        size_t ido = pattern[0][0];
        size_t idx = pattern[0][1] + itm->idx;
        size_t idy = pattern[0][2] + itm->idy;
        size_t idz = pattern[0][3];
        std::vector<double> p;
        p.push_back(bounds_[0].first + idx * step_size_);
        p.push_back(bounds_[1].first + idy * step_size_);
        p.push_back(bounds_[2].first + idz * step_size_);
        arma::mat r = rotZMat(-(*it));
        std::vector<double> base = wp - transFromMat(r * transMat(p));
        std::vector<double> quat = quatFromMat(r);
        base.insert(base.end(), quat.begin(), quat.end());
        // Assuming fixed z value
        if(fabs(base[2]-zbound.first) > 0.5 * step_size_) {
          std::cout << "Incorrect z value!" << std::endl;
        }
        base[2] = zbound.first;
        Placement placement;
        placement.base = base;
        placement.score = itm->score;
        placement.paths.push_back(path);
        placements.push_back(placement);
      }
    }
    return placements;
  }
  MultiPattern getMultiPathPattern(const PathVec& pathvec,
                                   const Pair& zbound,
                                   double rotz) {
    PathVec npathvec;
    arma::mat mrotz = rotZMat(rotz);
    for(auto it1 = pathvec.begin(); it1 != pathvec.end(); ++it1) {
      Path npath;
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        npath.push_back(transformFromMat(mrotz * transformMat(*it2)));
      }
      npathvec.push_back(npath);
    }
    return getMultiPathPattern(npathvec, zbound);
  }
  MultiPattern getMultiPathPattern(const PathVec& pathvec,
                                   const Pair& zbound) {
    MultiPattern multipattern;
    double xmin, ymin, zmin;
    xmin = ymin = 1e9;
    zmin = bounds_[2].first + zbound.first;
    for(auto it1 = pathvec.begin(); it1 != pathvec.end(); ++it1) {
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        if((*it2)[0] < xmin) xmin = (*it2)[0];
        if((*it2)[1] < ymin) ymin = (*it2)[1];
      }
    }
    // Make the first point match exactly
    double xoffset = step_size_ * (fmod(0.5 + (pathvec[0][0][0] - xmin) / step_size_, 1.0) - 1.5);
    double yoffset = step_size_ * (fmod(0.5 + (pathvec[0][0][1] - ymin) / step_size_, 1.0) - 1.5);
    xmin += xoffset;
    ymin += yoffset;
    for(auto it1 = pathvec.begin(); it1 != pathvec.end(); ++it1) {
      Pattern pattern;
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        PatternElement pi;
        pi.push_back(getClosestOriIdx(std::vector<double>(it2->begin()+3, it2->end())));
        pi.push_back(size_t(0.5 + ((*it2)[0] - xmin) / step_size_));
        pi.push_back(size_t(0.5 + ((*it2)[1] - ymin) / step_size_));
        pi.push_back(size_t(0.5 + ((*it2)[2] - zmin) / step_size_));
        pattern.push_back(pi);
      }
      multipattern.push_back(pattern);
    }
    return multipattern;
  }
  Matches searchMultiPattern(const MultiPattern& multipattern) {
    int idxmax = 0;
    int idymax = 0;
    for(auto it1 = multipattern.begin(); it1 != multipattern.end(); ++it1) {
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        if((*it2)[1] > idxmax) idxmax = (*it2)[1];
        if((*it2)[2] > idymax) idymax = (*it2)[2];
      }
    }
    Matches matches;
    for(int idx = -idxmax; idx < int(num_x_); ++idx) {
      for(int idy = -idymax; idy < int(num_y_); ++idy) {
        Match match;
        std::vector<ScoreT> scores;
        for(auto it1 = multipattern.begin(); it1 != multipattern.end(); ++it1) {
          ScoreT score = scorePattern(*it1, idx, idy, 0);
          if(score > 0) {
            scores.push_back(score);
            match.path_ids.push_back(it1 - multipattern.begin());
          }
        }
        match.score = std::accumulate(scores.begin(), scores.end(), 0.0);
        match.idx = idx;
        match.idy = idy;
        if(match.score > 0) {
          matches.push_back(match);
        }
      }
    }
    std::sort(matches.begin(), matches.end(), 
              [](const Match& a, const Match& b) { return a.score > b.score; });
    //return Matches(matches.begin(), matches.begin() + std::min(size_t(10), matches.size()));
    return matches;
  }
  Placements getMultiPathPlacements(const PathVec& pathvec,
                                    const Pair& zbound,
                                    const Pair& rbound,
                                    const arma::mat& extension) {
    arma::mat ext_inv = arma::inv(extension);
    PathVec npathvec;
    for(auto it1 = pathvec.begin(); it1 != pathvec.end(); ++it1) {
      Path p;
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        p.push_back(transformFromMat(transformMat(*it2) * ext_inv));
      }
      npathvec.push_back(p);
    }
    return getMultiPathPlacements(npathvec, zbound, rbound);
  }
  Placements getMultiPathPlacements(const PathVec& pathvec,
                                    const Pair& zbound,
                                    const Pair& rbound) {
    assert(zbound.first == zbound.second); // For now, only support fixed z value
    assert(rbound.first <  rbound.second); // Require a valid base z rot range, assuming upright
    Placements placements;
    std::vector<double> rotzs = linspace<double>(rbound.first, rbound.second, 10, false);
    for(auto it = rotzs.begin(); it != rotzs.end(); ++it) {
      MultiPattern multipattern = getMultiPathPattern(pathvec, zbound, *it);
      Matches matches = searchMultiPattern(multipattern);
      for(auto itm = matches.begin(); itm != matches.end(); ++itm) {
        std::vector<double> wp = std::vector<double>(pathvec[0][0].begin(), pathvec[0][0].begin()+3);
        size_t ido = multipattern[0][0][0];
        int idx = multipattern[0][0][1] + itm->idx;
        int idy = multipattern[0][0][2] + itm->idy;
        int idz = multipattern[0][0][3];
        std::vector<double> p;
        p.push_back(bounds_[0].first + idx * step_size_);
        p.push_back(bounds_[1].first + idy * step_size_);
        p.push_back(bounds_[2].first + idz * step_size_);
        arma::mat r = rotZMat(-(*it));
        std::vector<double> base = wp - transFromMat(r * transMat(p));
        std::vector<double> quat = quatFromMat(r);
        base.insert(base.end(), quat.begin(), quat.end());
        // Assuming fixed z value
        if(fabs(base[2]-zbound.first) > 0.5 * step_size_) {
          std::cout << "Incorrect z value!" << std::endl;
        }
        base[2] = zbound.first;
        Placement placement;
        placement.base = base;
        placement.score = itm->score;
        for(auto itp = itm->path_ids.begin(); itp != itm->path_ids.end(); ++itp) {
          placement.paths.push_back(pathvec[*itp]);
        }
        placements.push_back(placement);
      }
    }
    return placements;
  }
  std::vector<ScoreT> getMultiPathScores(const PathVec& pathvec,
                                         const Pose& placement,
                                         const arma::mat& extension) {
    arma::mat ext_inv = arma::inv(extension);
    PathVec npathvec;
    for(auto it1 = pathvec.begin(); it1 != pathvec.end(); ++it1) {
      Path p;
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        p.push_back(transformFromMat(transformMat(*it2) * ext_inv));
      }
      npathvec.push_back(p);
    }
    return getMultiPathScores(npathvec, placement);
  }
  std::vector<ScoreT> getMultiPathScores(const PathVec& pathvec,
                                         const Pose& placement) {
    std::vector<ScoreT> scores;
    arma::mat b2w = arma::inv(transformMat(placement));
    for(auto it1 = pathvec.begin(); it1 != pathvec.end(); ++it1) {
      Pattern pat;
      for(auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
        Pose tf = transformFromMat(b2w * transformMat(*it2));
        size_t ido = getClosestOriIdx(std::vector<double>(tf.begin()+3, tf.end()));
        PatternElement e;
        e.push_back(ido);
        e.push_back(getXIdx(tf[0]));
        e.push_back(getYIdx(tf[1]));
        e.push_back(getZIdx(tf[2]));
        pat.push_back(e);
      }
      scores.push_back(scorePattern(pat, 0, 0, 0));
    }
    return scores;
  }

private:
  std::string model_name_;
  std::vector<arma::mat> oris_;
  std::shared_ptr<KDTreeT> quats_index_;
  std::vector<std::vector<double> > quats_;
  std::vector<std::vector<ScoreT> > elems_;
  std::vector<Pair> bounds_;
  std::vector<double> midpt_;
  std::vector<double> halfr_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  double step_size_;
  size_t num_x_;
  size_t num_y_;
  size_t num_z_;
  size_t num_d_;
  size_t num_r_;
  size_t num_p_;
  size_t num_o_;
  size_t num_f_;
};

class BasePlanner {
public:
  BasePlanner(const std::string& model_name) {
    rdb_.reset(new ReachabilityDB(model_name, "/home/robotics/.reachdb/" + model_name)); 
  }
  ~BasePlanner() {}

public:
  Placements plan(const PathVec& paths,
                  const Pair& zbound,
                  const Pair& rbound,
                  const arma::mat& ext) {
    return rdb_->getMultiPathPlacements(paths, zbound, rbound, ext);
  }

  Placements pyplan(bp::object paths,
                    bp::object zbd,
                    bp::object rbd,
                    bp::object ext) {
    double zl = bp::extract<double>(zbd[0]);
    double zu = bp::extract<double>(zbd[1]);
    double rl = bp::extract<double>(rbd[0]);
    double ru = bp::extract<double>(rbd[1]);
    return plan(readPaths(paths), Pair(zl, zu), Pair(rl, ru), readMat(ext));
  }

  std::vector<ScoreT> score(const std::vector<double>& xs,
                            const std::vector<double>& ys,
                            const std::vector<double>& zs,
                            const std::vector<double>& q) {
    std::vector<ScoreT> scores;
    size_t ido = rdb_->getClosestOriIdx(q);
    for(auto itx = xs.begin(); itx != xs.end(); ++itx) {
      size_t idx = rdb_->getXIdx(*itx);
      for(auto ity = ys.begin(); ity != ys.end(); ++ity) {
        size_t idy = rdb_->getYIdx(*ity);
        for(auto itz = zs.begin(); itz != zs.end(); ++itz) {
          size_t idz = rdb_->getZIdx(*itz);
          scores.push_back(rdb_->getReachabilityElement(ido, idx, idy, idz));
        }
      }
    }
    return scores;
  }

  std::vector<ScoreT> pyscore(bp::object xs,
                              bp::object ys,
                              bp::object zs,
                              bp::object quat) {
    std::vector<double> xss, yss, zss, q;
    for(size_t i = 0; i < bp::len(xs); ++i) {
      xss.push_back(bp::extract<double>(xs[i]));
    }
    for(size_t i = 0; i < bp::len(ys); ++i) {
      yss.push_back(bp::extract<double>(ys[i]));
    }
    for(size_t i = 0; i < bp::len(zs); ++i) {
      zss.push_back(bp::extract<double>(zs[i]));
    }
    for(size_t i = 0; i < bp::len(quat); ++i) {
      q.push_back(bp::extract<double>(quat[i]));
    }
    return score(xss, yss, zss, q);
  }

  std::vector<ScoreT> rate(const PathVec& pathvec,
                           const Pose& placement,
                           const arma::mat& ext) {
    return rdb_->getMultiPathScores(pathvec, placement, ext);
  }

  std::vector<ScoreT> pyrate(bp::object paths,
                             bp::object placement,
                             bp::object ext) {
    Pose base;
    for(size_t i = 0; i < bp::len(placement); ++i) {
      base.push_back(bp::extract<double>(placement[i]));
    }
    return rate(readPaths(paths), base, readMat(ext));
  }

private:
  arma::mat readMat(bp::object pymat) {
    arma::mat mat = arma::zeros(4, 4);
    for(size_t i = 0; i < 4; i++) {
      for(size_t j = 0; j < 4; j++) {
        mat(i, j) = bp::extract<double>(pymat[i][j]);
      }
    }
    return mat;
  }
  PathVec readPaths(bp::object pypaths) {
    PathVec paths;
    for(size_t i = 0; i < bp::len(pypaths); i++) {
      Path path;
      for(size_t j = 0; j < bp::len(pypaths[i]); j++) {
        Pose p;
        for(size_t k = 0; k < bp::len(pypaths[i][j]); k++) {
          p.push_back(bp::extract<double>(pypaths[i][j][k]));
        }
        path.push_back(p);
      }
      paths.push_back(path);
    }
    return paths;
  }

private:
  std::shared_ptr<ReachabilityDB> rdb_;
};

template < class T > 
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "["; 
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii) {
        os << " " << *ii;
    } 
    os << " ]";
    return os;
}

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    std::vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::plus<T>());
    return result;
}

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    std::vector<T> result;
    result.reserve(a.size());
    std::transform(a.begin(), a.end(), b.begin(), 
                   std::back_inserter(result), std::minus<T>());
    return result;
}

template <typename T>
std::vector<T> operator*(const std::vector<T>& a, double b) {
    std::vector<T> result(a.size());
    std::transform(a.begin(), a.end(), result.begin(), 
                   std::bind1st(std::multiplies<T>(), b));
    return result;
}

template <typename T = double>
std::vector<T> linspace(T a, T b, size_t N, bool include_end) {
  T h;
  if(include_end) {
    h = (b - a) / static_cast<T>(N-1);
  }
  else {
    h = (b - a) / static_cast<T>(N);
  }
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

template<class T>
struct vec2py
{
  static PyObject* convert(const std::vector<T>& vec)
  {
    bp::list* l = new bp::list();
    for(std::size_t i = 0; i < vec.size(); i++)
      (*l).append(vec[i]);

    return l->ptr();
  }
};

BOOST_PYTHON_MODULE(bplanner) {
  bp::to_python_converter<std::vector<Placement, std::allocator<Placement> >, vec2py<Placement> >();
  bp::to_python_converter<std::vector<double, std::allocator<double> >, vec2py<double> >();
  bp::to_python_converter<std::vector<float, std::allocator<float> >, vec2py<float> >();
  bp::to_python_converter<std::vector<Path, std::allocator<Path> >, vec2py<Path> >();
  bp::to_python_converter<std::vector<Pose, std::allocator<Pose> >, vec2py<Pose> >();
  bp::class_<Placement>("Placement", bp::no_init)
    .def("get_base",  &Placement::getBase)
    .def("get_paths", &Placement::getPaths)
    .def("get_score", &Placement::getScore)
  ;
  bp::class_<BasePlanner, boost::noncopyable>("BasePlanner", bp::init<const std::string&>())
    .def("plan", &BasePlanner::pyplan)
    .def("rate", &BasePlanner::pyrate)
    .def("score", &BasePlanner::pyscore)
  ;
}
