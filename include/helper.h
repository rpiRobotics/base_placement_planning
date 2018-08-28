#include <cmath>
#include <armadillo>
#include <boost/python.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>

arma::mat rotXMat(double angle);
arma::mat rotYMat(double angle);
arma::mat rotZMat(double angle);
arma::mat eulerMat(double x, double y, double z);

arma::mat transMat(double x, double y, double z);
arma::mat transMat(const std::vector<double>& trans);
arma::mat quatMat(double x, double y, double z, double w);
arma::mat quatMat(const std::vector<double>& quat);
arma::mat transformMat(const std::vector<double>& transform);
std::vector<double> transFromMat(const arma::mat& mat);
std::vector<double> quatFromMat(const arma::mat& mat);
std::vector<double> transformFromMat(const arma::mat& mat);

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v);

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b);

template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b);

template <typename T>
std::vector<T> operator*(const std::vector<T>& a, double b);

template <typename T = double>
std::vector<T> linspace(T a, T b, size_t N, bool include_end=true);

typedef std::vector<double> VState;
typedef std::vector<VState> VStateVec;
typedef struct NodeInfo { 
  VState state; 
  friend std::ostream &operator<<(std::ostream &os, NodeInfo ni) { return os << ni.state; }
} NodeInfo;
typedef boost::property<boost::edge_weight_t, double> EdgeInfo;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, NodeInfo, EdgeInfo> GraphT;
typedef boost::graph_traits<GraphT>::vertex_descriptor NodeT;
typedef boost::graph_traits<GraphT>::edge_descriptor EdgeT;
typedef std::vector<GraphT::edge_descriptor> PathT;
typedef boost::property_map < GraphT, boost::vertex_index_t >::type IndexMap;
typedef boost::iterator_property_map < NodeT*, IndexMap, NodeT, NodeT& > PredecessorMap;
typedef boost::iterator_property_map < double*, IndexMap, double, double& > DistanceMap;

typedef bg::model::point<float, 7, bg::cs::cartesian> PointT;
typedef bg::model::box<PointT> BoxT;
typedef std::pair<BoxT, NodeT> ValueT;
typedef std::vector<ValueT> ValueVecT;
typedef bgi::rtree<ValueT, bgi::quadratic<16> > RTreeT;

namespace baxter_l_arm {
VState solvefk(const VState& joints);
VStateVec solveik(const VState& mvalues, double free);
VStateVec solveik(const VState& mvalues, const VState& frees);
};
namespace baxter_r_arm {
VState solvefk(const VState& joints);
VStateVec solveik(const VState& mvalues, double free);
VStateVec solveik(const VState& mvalues, const VState& frees);
};

class BoostRTree {
public:
  BoostRTree(const VState& weights={5,5,5,1,1,1,1}) {
    setWeights(weights);
  }

  void setWeights(const VState& weights) {
    weights_ = weights;
  }

  void insert(const VState& vstate, const NodeT& n) {
    PointT p = getPoint(vstate);
    rtree_impl_.insert(std::make_pair(BoxT(p, p), n));
  }

  ValueVecT getKNN(const VState& vstate, size_t k) {
    ValueVecT results;
    PointT p = getPoint(vstate);
    rtree_impl_.query(bgi::nearest(p, k), std::back_inserter(results));
    return results;
  }

  VState getVState(const PointT& point) {
    VState vstate(7);
    vstate[0] = bg::get<0>(point);
    vstate[1] = bg::get<1>(point);
    vstate[2] = bg::get<2>(point);
    vstate[3] = bg::get<3>(point);
    vstate[4] = bg::get<4>(point);
    vstate[5] = bg::get<5>(point);
    vstate[6] = bg::get<6>(point);
    return vstate;
  }

  PointT getPoint(const VState& vstate) {
    PointT point;
    bg::set<0>(point, vstate[0]);
    bg::set<1>(point, vstate[1]);
    bg::set<2>(point, vstate[2]);
    bg::set<3>(point, vstate[3]);
    bg::set<4>(point, vstate[4]);
    bg::set<5>(point, vstate[5]);
    bg::set<6>(point, vstate[6]);
    return point;
  }

private:
  RTreeT rtree_impl_;
  VState weights_;
};

class RTreeGraph {
public:

  RTreeGraph(void) {}
  virtual ~RTreeGraph(void) {}

  NodeT addNode(const VState& state, bool invisible=false) {
    NodeT n = boost::add_vertex(graph_);
    graph_[n].state = state;
    if(!invisible) {
      rtree_.insert(state, n);
    }
    return n;
  }

  void addEdge(NodeT a, NodeT b, double weight=1) {
    boost::add_edge(a, b, weight, graph_);
  }

  void delEdge(NodeT a, NodeT b) {
    boost::remove_edge(a, b, graph_);
  }

  void safeEdge(NodeT a, NodeT b) {
    if(a < b) {
      safe_edges_.insert(std::pair<NodeT, NodeT>(a, b));
    }
    else {
      safe_edges_.insert(std::pair<NodeT, NodeT>(b, a));
    }
  }

  bool isSafeEdge(NodeT a, NodeT b) {
    if(a < b) {
      return safe_edges_.find(std::pair<NodeT, NodeT>(a, b)) != safe_edges_.end();
    }
    else {
      return safe_edges_.find(std::pair<NodeT, NodeT>(b, a)) != safe_edges_.end();
    }
  }

  bool hasEdge(NodeT a, NodeT b) {
    return boost::edge(a, b, graph_).second;
  }

  size_t numNodes() {
    return boost::num_vertices(graph_);
  }

  size_t numEdges() {
    return boost::num_edges(graph_);
  }

  VState getNodeVState(NodeT node) {
    return graph_[node].state;
  }

  VStateVec getNodeStates(const std::vector<NodeT>& nodes) {
    VStateVec path;
    for(auto it = nodes.begin(); it != nodes.end(); ++it) {
      path.push_back(graph_[*it].state);
    }
    return path;
  }

  std::vector<NodeT> dijkstra(NodeT s, NodeT dst) {
    std::vector<NodeT> predecessors(boost::num_vertices(graph_));
    std::vector<double> distances(boost::num_vertices(graph_));
     
    IndexMap indexMap = boost::get(boost::vertex_index, graph_);
    PredecessorMap pMap(&predecessors[0], indexMap);
    DistanceMap dMap(&distances[0], indexMap);
     
    boost::dijkstra_shortest_paths(graph_, s, boost::distance_map(dMap).predecessor_map(pMap));
    std::vector<NodeT> path;
    NodeT v = dst;
    for(NodeT u = pMap[v]; u != v; v = u, u = pMap[v]) {
      auto source = boost::source(boost::edge(u, v, graph_).first, graph_);
      path.push_back(source);
    }
    return path;
  }
 
  void print() {
    boost::write_graphviz(std::cout, graph_);
  }

  ValueVecT getKNN(const VState& vstate, size_t k) {
    return rtree_.getKNN(vstate, k);
  }

  VState getVState(const PointT& point) {
    return rtree_.getVState(point);
  }

private:
  GraphT graph_;
  BoostRTree rtree_;
  std::set<std::pair<NodeT, NodeT> > safe_edges_;
};
