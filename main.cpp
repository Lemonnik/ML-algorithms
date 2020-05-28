#include <iostream>
#include <armadillo>
#include "split.h"
#include "defs.h"
#include "metrics.h"
#include "linreg.h"


int main() {
	arma::mat X, y, y_pred, y_test;
	arma::uvec r, except;
	MSE m = MSE();
	LinearRegression l;
	X.load("insurance.txt", arma::csv_ascii);
	y = Split::splitTarget(X, X.n_cols - 1);


	/* single train-test split */
	defs::Pair p;
	Split::trainTestSplit(0.8, X.n_rows, p);

	except = {2, 3, 4, 5, 6, 7};

	/* analytical solution with normal equations */
	l = LinearRegression(X.rows(p.first), y.rows(p.first), true, except, &m);
	y_pred = l.predict(X.rows(p.second), y.rows(p.second), except, true);

	/* numerical solution with gradient descent */
	l = LinearRegression(X.rows(p.first), y.rows(p.first), false, except, &m);
	y_pred = l.predict(X.rows(p.second), y.rows(p.second), except, false);

	/* cross-validation with 5 folds */
	/*defs::Pair p1;
	Split::cvSplit(5, X.n_rows, p1);
	for (int i = 0; i < 5; i++) {
		r = p1.first.col(i);
		l = LinearRegression(X.rows(r), y.rows(r), false, &m);
		r = p1.second.col(i);
		l.predict(X.rows(r), y.rows(r), false);
	}*/
}
