#include "helper.h"

arma::mat rotXMat(double angle) {
  return {
    { 1,           0,          0, 0 },   
    { 0,  cos(angle), sin(angle), 0 },
    { 0, -sin(angle), cos(angle), 0 },
    { 0,           0,          0, 1 }
  };
}

arma::mat rotYMat(double angle) {
  return {
    {  cos(angle), 0, -sin(angle), 0 },
    {           0, 1,           0, 0 },
    {  sin(angle), 0,  cos(angle), 0 },
    {           0, 0,           0, 1 }
  };
}

arma::mat rotZMat(double angle) {
  return {
    {  cos(angle), sin(angle), 0, 0 },
    { -sin(angle), cos(angle), 0, 0 },
    {           0,          0, 1, 0 },
    {           0,          0, 0, 1 }
  };
}

arma::mat eulerMat(double x, double y, double z) {
  return rotZMat(z) * rotYMat(y) * rotXMat(x);
}

arma::mat transMat(double x, double y, double z) {
  return {
    {1, 0, 0, x},
    {0, 1, 0, y},
    {0, 0, 1, z},
    {0, 0, 0, 1}
  };
}

arma::mat transMat(const std::vector<double>& trans) {
  return transMat(trans[0], trans[1], trans[2]);
}

arma::mat quatMat(double x, double y, double z, double w) {
  double nm = sqrt(x*x + y*y + z*z + w*w);
  x = x / nm;
  y = y / nm;
  z = z / nm;
  w = w / nm;
  return {
    {1-2*y*y-2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0},
    {2*x*y + 2*z*w, 1-2*x*x-2*z*z, 2*y*z - 2*x*w, 0},
    {2*x*z - 2*y*w, 2*y*z + 2*x*w, 1-2*x*x-2*y*y, 0},
    {0, 0, 0, 1}
  };
}

arma::mat quatMat(const std::vector<double>& quat) {
  return quatMat(quat[0], quat[1], quat[2], quat[3]);
}

arma::mat transformMat(const std::vector<double>& transform) {
  return transMat(std::vector<double>(transform.begin(), transform.begin()+3))
        * quatMat(std::vector<double>(transform.begin()+3, transform.end()));
}

std::vector<double> transFromMat(const arma::mat& mat) {
  return std::vector<double>({mat(0, 3), mat(1, 3), mat(2, 3)});
}

std::vector<double> quatFromMat(const arma::mat& mat) {
  double x, y, z, w;
  double trace = mat(0, 0) + mat(1, 1) + mat(2, 2);
  if( trace > 0 ) {
    double s = 0.5 / sqrtf(trace+ 1.0);
    w = 0.25 / s;
    x = ( mat(2, 1) - mat(1, 2) ) * s;
    y = ( mat(0, 2) - mat(2, 0) ) * s;
    z = ( mat(1, 0) - mat(0, 1) ) * s;
  } else {
    if ( mat(0, 0) > mat(1, 1) && mat(0, 0) > mat(2, 2) ) {
      double s = 2.0 * sqrtf( 1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2));
      w = (mat(2, 1) - mat(1, 2) ) / s;
      x = 0.25 * s;
      y = (mat(0, 1) + mat(1, 0) ) / s;
      z = (mat(0, 2) + mat(2, 0) ) / s;
    } else if (mat(1, 1) > mat(2, 2)) {
      double s = 2.0 * sqrtf( 1.0 + mat(1, 1) - mat(0, 0) - mat(2, 2));
      w = (mat(0, 2) - mat(2, 0) ) / s;
      x = (mat(0, 1) + mat(1, 0) ) / s;
      y = 0.25 * s;
      z = (mat(1, 2) + mat(2, 1) ) / s;
    } else {
      double s = 2.0 * sqrtf( 1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1) );
      w = (mat(1, 0) - mat(0, 1) ) / s;
      x = (mat(0, 2) + mat(2, 0) ) / s;
      y = (mat(1, 2) + mat(2, 1) ) / s;
      z = 0.25 * s;
    }
  }
  if(w < 0) {
    x *= -1;
    y *= -1;
    z *= -1;
    w *= -1;
  }
  return std::vector<double>({x, y, z, w});
}

std::vector<double> transformFromMat(const arma::mat& mat) {
  std::vector<double> transform = transFromMat(mat);
  std::vector<double> quat = quatFromMat(mat);
  transform.insert(transform.end(), quat.begin(), quat.end());
  return transform;
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
