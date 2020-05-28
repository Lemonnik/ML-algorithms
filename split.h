#ifndef SPLIT_H_
#define SPLIT_H_

#include <armadillo>
#include <utility>
#include "defs.h"

using namespace defs;

class Split {
public:
	static void trainTestSplit (double prop, ULL n, Pair &ind) {
		arma::uvec train_ind, test_ind;
		arma::uvec p = arma::randperm(n);

		ULL m = ULL(prop * n);

		ind.first = p.head(m);
		ind.second = p.tail(n - m);
	}

	static void cvSplit (int k, ULL n, Pair &ind) {
		ULL offset = 0LL, fold = ULL(n / k);

		arma::uvec p = arma::randperm(n), train, test;
		for (int i = 0; i < k; i++) {
			// one fold for test
			test = p.rows(offset, offset + fold - 1);
			// k-1 folds for train,
			// copy is need as shed_rows removes in-place
			train = p;
			train.shed_rows(offset, offset + fold - 1);
			offset += fold;
			ind.first.insert_cols(ind.first.n_cols, train);
			ind.second.insert_cols(ind.second.n_cols, test);
		}
	}

	static arma::mat splitTarget (arma::mat &data, ULL col) {
		arma::mat y = data.col(col);
		data.shed_col(col);
		return y;
	}
};
#endif /* SPLIT_H_ */
