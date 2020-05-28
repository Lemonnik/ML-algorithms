#include <armadillo>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "defs.h"
#include "metrics.h"

using namespace defs;

class LinearRegression {
private:
	/* coefficients, residuals and predictions */
	arma::mat theta, res, y_pred;
	/* training set mean and deviation */
	arma::mat mu, sd;
    /* learning rate and difference between cost f. values */
    double alpha, eps;
    /* maximum iterations of gradient descent */
    int max_it;
    /* cost function and its gradient */
    costFunc *F;
public:
    LinearRegression() {};
    LinearRegression(const arma::mat &X_, const arma::mat &y, bool norm_eq = true,
    		const arma::uvec &except=NULL, costFunc *f = NULL, double a = 0.001, int it = 1000, double e = 1.0e-20) {
        arma::mat X = X_; 
        F = f;
        if (norm_eq) {
            addOne(X);
            normalEqn(X, y);
        }
        else {
        	alpha = a;
        	max_it = it;
        	eps = e;
            featNormalize (X, except);
            addOne (X);
            gradientDescent (X, y);
        }
        saveCoefs ();
        modelSummary (X, y);
    }
    
	/* standardization */
	void featNormalize (arma::mat &X, const arma::uvec &c, bool mu_sd = true) {
        if (mu_sd) {
            mu = arma::mean (X, 0);
            mu.cols(c) = arma::zeros(c.n_rows).t();
            sd = arma::stddev (X, 0);
            sd.cols(c) = arma::ones(c.n_rows).t();
        }
		X.each_row([this](arma::rowvec &x) {x = (x - mu) / sd;});
	}

	void addOne (arma::mat &X) {
		X.insert_cols(0, arma::ones(X.n_rows));
	}

	void initCoefs (ULL n) {
		theta = 1. / (2 * std::sqrt(n)) * arma::ones(n);
    }

	arma::mat J (const arma::mat &X) {
		return X * theta;
	}

	double computeCost (const arma::mat &X, const arma::mat &y) {
		y_pred = J(X);
		res = y_pred - y;
		return F->loss(y, y_pred);
	}

	arma::mat getCoefs () {
		return theta;
	}

	void gradientDescent (const arma::mat &X, const arma::mat &y) {
		double prev_cost = 0., cur_cost;
        std::ofstream out;
        out.open("cost_history.out", std::ofstream::out | std::ofstream::trunc);
        initCoefs(X.n_cols);
		for (int i = 0; i < max_it; i++) {
			cur_cost = computeCost(X, y);
            out << i + 1 << "," << std::setprecision(20) << cur_cost << '\n';
			if (std::abs(prev_cost - cur_cost) < eps) {
				break;
			}
			theta = theta - alpha * F->grad(X, y, y_pred);
			if ((i + 100) % 100 == 0) {
				alpha *= 0.1;
			}
			prev_cost = cur_cost;
		}
		out.close();
	}

	void normalEqn (const arma::mat &X, const arma::mat &y) {
		theta = arma::inv(X.t() * X) * X.t() * y;
	}
	
	void saveCoefs () {
        std::ofstream out;
        out.open("coefficients.out", std::ofstream::out | std::ofstream::app);
        FILL(out);
        out << theta.t();
        FILL(out);
        out.close();
    }
	
	arma::mat predict (const arma::mat &X_test, const arma::mat &y_test,
			const arma::uvec &except, bool normEq = false) {
        arma::mat X = X_test;
        if (!normEq) {
        	featNormalize(X, except, false);
        }
        addOne(X);
        predictSummary(X, y_test);
        return y_pred;
    }
    
    void modelSummary (const arma::mat &X, const arma::mat &y) {
    	ULL n = X.n_rows, m = X.n_cols;
    	double cost, s[5];

        std::ofstream out;
        out.open("model_summary.out", std::ofstream::out | std::ofstream::app);

        cost = computeCost(X, y);
        out.precision(5);
        FILL(out);
        out << "RMSE train: " << std::setw(10) << std::sqrt(cost) << '\n';
        out << "R squared: " << std::setw(10) << Metrics::R2(y, y_pred) << '\n';
        out << "Adjusted R squared: " << std::setw(10) << Metrics::adjR2(n, m, y, y_pred) << '\n';

        Metrics::resSummary(res, s);
        const char *names[] = {"Min", "1Q", "Median", "3Q", "Max"};
        out << "Residuals:\n";

        for (int i = 0; i < 5; i++) {
        	out << std::setw(10) << names[i] << ": " << s[i] << '\n';
        }
        FILL(out);
        out.close();
    }

    void predictSummary (const arma::mat &X_test, const arma::mat &y_test) {
    	std::ofstream out;
    	out.open("predict_summary.out", std::ofstream::out | std::ofstream::app);

    	double cost = computeCost(X_test, y_test);
    	double mape = Metrics::MAPE(y_test, y_pred);
    	arma::mat c = arma::cor(y_test, y_pred);


    	out.precision(5);
    	FILL(out);
    	out << "Correlation coef. between observations and predictions: "
    			<< std::setw(10) << c(0) << '\n';

    	out << "RMSE: " << std::setw(10) << std::sqrt(cost) << '\n';
    	out << "MAPE: " << std::setw(10) << mape << "%\n";

    	FILL(out);
    	out.close();
    }
}; 
