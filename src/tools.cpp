#include <iostream>
#include "tools.h"
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	
	// Initialize RMSE as vector of 4 numbers
	// x,y,vx,vy
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;
	size_t N = estimations.size();

	// Check that estimation size is not 0
	if (N == 0) {
		std::cout << "error: estimation size is 0";
		return rmse;
	}

	// Ensure estimation size is equal to ground truth size
	if (N != ground_truth.size()) {
		std::cout << "error: esimation size not equal to ground truth size";
		return rmse;
	}

	// Calculate residual, square it, then accumulate sum
	for (size_t i = 0; i < N; ++i) {
		VectorXd est = estimations[i];
		VectorXd gnd = ground_truth[i];

		VectorXd res = est - gnd;
		VectorXd res_sq = res.array()*res.array();

		rmse += res_sq;
	}

	// Get mean of sum of squared residuals
	rmse = rmse / N;

	// Take square root
	rmse = rmse.array().sqrt();

	return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	
	// Initialize output
	// 3 rows for rho, phi, and rho_dot
	// 4 columns for px,py,vx,vy
	MatrixXd Hj(3, 4);

	// Recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);

	// Calculate often-used parameters
	double rho = sqrt(px*px + py*py);
	double rho_sq = rho*rho;
	double rho_cube = rho_sq*rho;

	// Check division by zero
	if (rho == 0) {
		Hj << 0, 0, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0;
		//std::cout << "CalculateJacobianError() - divide by zero";		
	}
	else {
		// Compute Jacobian matrix
		Hj << px/rho,                         py/rho,                       0,        0,
					-py/rho_sq,                     px/rho_sq,                    0,        0,
					py*(vx*py - vy*px) / rho_cube, px*(vy*px - vx*py) / rho_cube, px / rho, py / rho;
	}

	return Hj;
	
}
