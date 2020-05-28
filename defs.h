#ifndef DEFS_H_
#define DEFS_H_

#include <utility>
#include <armadillo>

#define FILL(out) out << std::setw(100) << std::setfill('*') << " \n";

namespace defs {
	typedef unsigned long long ULL;
	typedef std::pair<arma::umat, arma::umat> Pair;
}

#endif /* DEFS_H_ */
