#include <vlfd_experiments/KDTreeVectorOfVectorsAdaptor.h>
#include <vlfd_experiments/helper.h>
#include <algorithm>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <ros/ros.h>
#include <Eigen/SVD>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl_parser/kdl_parser.hpp>

using namespace std::chrono;

high_resolution_clock::time_point t1, t2;
duration<double, std::milli> dur;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

typedef KDTreeVectorOfVectorsAdaptor<VStateVec, double>  KDTreeT;
typedef std::pair<double, double> Pair;
typedef float ScoreT;


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
    setupKDL("left_arm_mount", "left_gripper_base");
    if(!load(filename)) {
      setParameters();
    }
    build(filename);
  }
  ~ReachabilityDB() {}

public:
  bool setupKDL(const std::string& base, const std::string& tip) {
    KDL::Tree kdltree;
    std::string robot_desc_string;
    ros::NodeHandle node;
    node.param("robot_description", robot_desc_string, std::string());
    if (!kdl_parser::treeFromString(robot_desc_string, kdltree)){
        return false;
    }
    KDL::Chain chain;
    kdltree.getChain(base, tip, chain);
    jac_.reset(new KDL::Jacobian(chain.getNrOfJoints()));
    jac_solver_.reset(new KDL::ChainJntToJacSolver(chain));
    svd_.reset(new Eigen::JacobiSVD<Eigen::MatrixXd>(jac_->data));
    return true;
  }
  arma::mat solveFK(const VState& joints) {
    if(model_name_ == "baxter_arm") {
      return arma::reshape(arma::mat(baxter_l_arm::solvefk(joints)), 4, 4).t();
    }
  }
  void build(const std::string& filename) {

    std::cout << "BUILDING REACHABILITY DATABASE.." << std::endl;

    // Generate database
    size_t batch = 100000000;
    VState joints(7);
    VState diffe = upper_ - lower_;
    for(size_t b = 0; b < 10; ++b) {
      t1 = high_resolution_clock::now();
      for(size_t i = 0; i < batch; ++i) {
        for(size_t j = 0; j < 7; ++j) {
          joints[j] = lower_[j] + (diffe[j] * dis(gen));
        }
        VState tf = transformFromMat(solveFK(joints));
        if(tf[0] < bounds_[0].first || tf[0] > bounds_[0].second) continue;
        if(tf[1] < bounds_[1].first || tf[1] > bounds_[1].second) continue;
        if(tf[2] < bounds_[2].first || tf[2] > bounds_[2].second) continue;
        VState quat(tf.begin()+3, tf.end());
        size_t ido = getClosestOriIdx(quat);
        size_t idp = getPosIdx(getXIdx(tf[0]), getYIdx(tf[1]), getZIdx(tf[2]));
        ScoreT score = scoreElem(joints);
        if(elems_[ido][idp] < score) {
          elems_[ido][idp] = score;
        }
      }
      num_f_ += batch;
      dur = high_resolution_clock::now() - t1;
      std::cout << "TIME (ms): " << dur.count() << std::endl;

      // Output statistics
      stat();

      // Save the database
      std::stringstream ss;
      ss << filename << (num_f_ / batch);
      t1 = high_resolution_clock::now();
      save(ss.str());
      dur = high_resolution_clock::now() - t1;
      std::cout << "TIME (ms): " << dur.count() << std::endl;
    }
  }
  ScoreT limit(const VState& joints) {
    ScoreT dist = 0;
    for(size_t i = 0; i < joints.size(); ++i) {
      ScoreT tmp = std::pow(1.2*(joints[i]-midpt_[i])/halfr_[i], 10);
      if(tmp > dist) dist = tmp;
    }
    return 0.1 / (dist+0.1);
  }
  ScoreT manip(const VState& joints) {
    KDL::JntArray jarr(joints.size());
    for(size_t i = 0; i < jarr.rows(); i++) {
        jarr(i) = joints[i];
    }
    jac_solver_->JntToJac(jarr, *jac_);
    svd_->compute(jac_->data);
    ScoreT cd = svd_->singularValues()(0) / svd_->singularValues()(svd_->singularValues().size()-1);
    return 1.0 / sqrt(cd);
  }
  ScoreT scoreElem(const VState& joints) {
    return limit(joints) * manip(joints);
  }
  void save(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    for(auto itb = bounds_.begin(); itb != bounds_.end(); ++itb) {
      ofs.write(reinterpret_cast<char*>(&(itb->first)), sizeof(double));
      ofs.write(reinterpret_cast<char*>(&(itb->second)), sizeof(double));
    }
    ofs.write(reinterpret_cast<char*>(&step_size_), sizeof(double));
    ofs.write(reinterpret_cast<char*>(&num_x_), sizeof(size_t));
    ofs.write(reinterpret_cast<char*>(&num_y_), sizeof(size_t));
    ofs.write(reinterpret_cast<char*>(&num_z_), sizeof(size_t));
    ofs.write(reinterpret_cast<char*>(&num_d_), sizeof(size_t));
    ofs.write(reinterpret_cast<char*>(&num_r_), sizeof(size_t));
    ofs.write(reinterpret_cast<char*>(&num_f_), sizeof(size_t));
    for(auto itq = quats_.begin(); itq != quats_.end(); ++itq) {
      ofs.write(reinterpret_cast<char*>(&(*(itq->begin()))), itq->size() * sizeof(double));
    }
    for(auto it1 = elems_.begin(); it1 != elems_.end(); ++it1) {
      ofs.write(reinterpret_cast<char*>(&(*(it1->begin()))), it1->size() * sizeof(ScoreT));
    }
    ofs.close();
  }
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
                                          [](ScoreT e){ return e > 0.1; }) << std::endl;
    std::cout << "RCH: " << std::count_if(all.begin(), all.end(),
                                          [](ScoreT e){ return e > 0.5; }) << std::endl;
    std::cout << "SIZE int: "    << sizeof(int)    << std::endl;
    std::cout << "SIZE float: "  << sizeof(float)  << std::endl;
    std::cout << "SIZE size_t: " << sizeof(size_t) << std::endl;
    std::cout << "SIZE double: " << sizeof(double) << std::endl;
  }
  void setParameters() {
    bounds_.clear();
    bounds_.push_back(Pair(0.0, 1.2));
    bounds_.push_back(Pair(-1.2, 1.2));
    bounds_.push_back(Pair(-0.3, 0.7));
    step_size_ = 0.04;
    num_x_ = 1 + (bounds_[0].second - bounds_[0].first) / step_size_;
    num_y_ = 1 + (bounds_[1].second - bounds_[1].first) / step_size_;
    num_z_ = 1 + (bounds_[2].second - bounds_[2].first) / step_size_;
    num_d_ = 256;
    num_r_ = 16;
    num_o_ = num_d_ * num_r_;
    num_f_ = 0;

    // Generate orientations and their index
    t1 = high_resolution_clock::now();
    genOris(num_d_, num_r_);
    dur = high_resolution_clock::now() - t1;
    std::cout << "TIME (ms): " << dur.count() << std::endl;

    // Allocate memory
    t1 = high_resolution_clock::now();
    elems_.resize(oris_.size());
    for(auto it = elems_.begin(); it != elems_.end(); ++it) {
      it->resize(num_x_ * num_y_ * num_z_, 0);
    }
    dur = high_resolution_clock::now() - t1;
    std::cout << "TIME (ms): " << dur.count() << std::endl;
  }
  bool genOris(size_t num_d, size_t num_r) {
    std::vector<Pair> dirs;
    double h_k, theta_k, phi_k;
    for(size_t k = 1; k <= num_d; ++k) {
      h_k = -1 + 2.0 * (k-1) / (num_d-1);
      theta_k = acos(h_k);
      if(k == 1 || k == num_d) {
        phi_k = 0;
      }
      else {
        phi_k = fmod((dirs[dirs.size()-1].second + 3.6 / sqrt(num_d * (1-h_k*h_k))), (M_PI*2));
      }
      dirs.push_back(Pair(theta_k, phi_k));
    }
    oris_.clear();
    quats_.clear();
    std::vector<double> rots = linspace<double>(0, M_PI*2, num_r, false);
    for(auto pit = dirs.begin(); pit != dirs.end(); ++pit) {
      arma::mat mdir = rotZMat(pit->second) * rotXMat(pit->first);
      for(auto rit = rots.begin(); rit != rots.end(); ++rit) {
        oris_.push_back(mdir * rotZMat(*rit));
        quats_.push_back(quatFromMat(oris_[oris_.size()-1]));
      }
    }
    quats_index_.reset(new KDTreeT(4, quats_, 10));
    return true;
  }
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
  ScoreT getReachabilityElement(size_t ido, size_t idx, size_t idy, size_t idz) {
    if(ido >= num_o_ || idx >= num_x_ || idy >= num_y_ || idz >= num_z_) return 0;
    else return elems_[ido][getPosIdx(idx, idy, idz)];
  }

private:
  std::shared_ptr<KDL::Jacobian> jac_;
  std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver_;
  std::shared_ptr<Eigen::JacobiSVD<Eigen::MatrixXd> > svd_;
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

template < class T > 
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) {
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

int main(int argc, char** argv) {
  ros::init(argc, argv, "generator");
  std::string model_name = "baxter_arm";
  ReachabilityDB(model_name, "/home/robotics/.reachdb/" + model_name); 
}
