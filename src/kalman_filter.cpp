#include "kalman_filter.h"
#include "tools.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd I = MatrixXd::Identity(4, 4);
Tools tools;
VectorXd h_x = VectorXd(3);

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_laser_in, MatrixXd &R_radar_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_laser_ = R_laser_in;
	R_radar_ = R_radar_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	
	x_ = F_*x_;
	P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	// Update using linear H, for laser

	VectorXd y_ = z - H_*x_;
	MatrixXd Ht_ = H_.transpose();
	MatrixXd S_ = H_*P_*Ht_ + R_laser_;
	MatrixXd K_ = P_*Ht_*S_.inverse();

	// new state
	x_ = x_ + K_*y_;
	P_ = (I - K_*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

	// Same as Update but use Jacobian Hj instead of H
	// except when calculating y
	// For radar
	
	// Recover state parameters
	double px = x_(0);
	double py = x_(1);
	double vx = x_(2);
	double vy = x_(3);

	// Map state into measurement space
	// In this case, cartesian to polar coordinates
	double rho = sqrt(px*px + py*py);
	double phi = atan2(py,px);
	
	double rho_dot;
	if (rho == 0) {
		rho_dot = 0;
	}
	else {
		rho_dot = (px*vx + py*vy) / rho;
	}

	// Assemble mapped state
	// h_x represents result of h(x)
	// Called z_pred in the lessons
	
	h_x << rho, phi, rho_dot;

	
	std::cout << z;
	std::cout << h_x;

	VectorXd y_ = z - h_x;
	std::cout << y_;

	MatrixXd Hj_ = tools.CalculateJacobian(x_);
	MatrixXd Hjt_ = Hj_.transpose();
	MatrixXd S_ = Hj_*P_*Hjt_ + R_radar_;
	MatrixXd K_ = P_*Hjt_*S_.inverse();
	
	// new state	
	
	MatrixXd ky = K_*y_;
	x_ = x_ + K_*y_;
	P_ = (I - K_*Hj_)*P_;
}
