#ifndef METRICS_H_
#define METRICS_H_

#include <armadillo>
#include "defs.h"

using namespace defs;

class costFunc {
public:
	virtual ~costFunc() {};
	virtual double loss (const arma::mat &y_true, const arma::mat &y_pred)=0;
	virtual arma::mat grad (const arma::mat &X, const arma::mat &y_true,
			const arma::mat &y_pred)=0;
};

class MSE: public costFunc {
public:
	~MSE() {};
	double loss (const arma::mat &y_true, const arma::mat &y_pred) {
		ULL n = y_true.n_rows;
		arma::mat res = y_pred - y_true;
		arma::mat sum_err = res.t() * res;
		return 1. / (2. * n) * sum_err(0);
	}

	/* vectorized gradient of MSE */
	arma::mat grad (const arma::mat &X, const arma::mat &y_true,
			const arma::mat &y_pred) {
		return X.t() * (y_pred - y_true);
	}
};

class Metrics {
public:
	/* sum of squared errors */
	static double SSE (const arma::mat &y_true, const arma::mat &y_pred) {
		arma::mat res = y_true - y_pred;
		arma::mat sse = res.t() * res;
        return sse(0);
    }

	/* sum of squared total */
    static double SST (const arma::colvec &y) {
        arma::colvec sst = y - arma::mean(y);
        sst = sst.t() * sst;
        return sst(0);
    }

    static double R2 (const arma::mat &y_true, const arma::mat &y_pred) {
    	return 1 - SSE(y_true, y_pred) / SST(y_true);
    }

    static double adjR2 (ULL n, ULL m, const arma::mat &y_true, const arma::mat &y_pred) {
        return 1 - (1 - R2(y_true, y_pred)) * (n - 1) / (n - m);
    }

    static double MAPE (const arma::mat &y_true, const arma::mat &y_pred) {
    	arma::mat d = arma::abs((y_true - y_pred) / y_true);
    	ULL n = d.n_rows;
    	d = d.t() * arma::ones(n);
    	return 100.0 / n * d(0);
    }

	static void medianInd(ULL *idx, ULL n) {
		if (n % 2) {
			idx[0] = idx[1] = n / 2;
		}
		else {
			idx[0] = n / 2 - 1;
			idx[1] = n / 2;
		}
	}

	static void resSummary (const arma::mat &res, double *s) {
		arma::colvec sorted = res.col(0);
		sorted = arma::sort(sorted);
		ULL idx[2], n = res.n_rows;

		medianInd(idx, n);
		s[0] = sorted(0); //min
		s[1] = arma::median(sorted.head_rows(idx[0] + 1)); // 1st quartile
		s[2] = arma::median(sorted); // 2nd quartile (median)
		s[3] = arma::median(sorted.tail_rows(n - idx[1])); //3rd quartile
		s[4] = sorted(n - 1); //max
	}
};

#endif /* METRICS_H_ */
