//
// Created by ldx on 24-5-6.
//
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <random>
#include <Eigen/Geometry>

using namespace std;

void ePnP(const std::vector<Eigen::Vector3d> &p3ds, const std::vector<Eigen::Vector2d> &p2ds, const Eigen::Matrix3d &K, Eigen::Matrix3d &R,
          Eigen::Vector3d &T) {
    /**寻找4个控制点**/
    //求取重心
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> control_points_w(4);
    for (int i = 0; i < p3ds.size(); i++) {
        center(0) += p3ds[i](0);
        center(1) += p3ds[i](1);
        center(2) += p3ds[i](2);
    }
    center(0) /= p3ds.size();
    center(1) /= p3ds.size();
    center(2) /= p3ds.size();
    control_points_w[0] = center;
    //构建矩阵
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p3ds.size(), 3);
    for (int i = 0; i < p3ds.size(); i++) {
        A(i, 0) = p3ds[i](0) - center(0);
        A(i, 1) = p3ds[i](1) - center(1);
        A(i, 2) = p3ds[i](2) - center(2);
    }

    Eigen::MatrixXd M = A.transpose() * A;

    Eigen::EigenSolver<Eigen::MatrixXd> solver(M);
    Eigen::VectorXd eigenValues = solver.eigenvalues().real();
    Eigen::MatrixXd eigenVectors = solver.eigenvectors().real();

    for (int i = 1; i < 4; i++) {
        control_points_w[i] = control_points_w[0] + sqrt(eigenValues(i - 1)) * eigenVectors.col(i - 1);
    }

    /**求解alpha**/
    Eigen::MatrixXd C_w = Eigen::MatrixXd::Zero(4, 4);
    for (int i = 0; i < 4; i++) {
        double x = control_points_w[i](0);
        double y = control_points_w[i](1);
        double z = control_points_w[i](2);
        C_w(0, i) = x;
        C_w(1, i) = y;
        C_w(2, i) = z;
        C_w(3, i) = 1;
    }
    Eigen::MatrixXd C_w_inv = C_w.inverse();


    double fu = K(0, 0);
    double fv = K(1, 1);
    double uc = K(0, 2);
    double vc = K(1, 2);

    std::vector<Eigen::Vector4d> alphas;
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(int(2 * p3ds.size()), 12);
    for (int i = 0; i < p3ds.size(); i++) {
        Eigen::Vector3d p3d = p3ds[i];
        Eigen::Vector2d p2d = p2ds[i];
        Eigen::Vector4d b;
        b << p3d(0), p3d(1), p3d(2), 1;
        Eigen::Vector4d alpha = C_w_inv * b;
        alphas.push_back(alpha);

        D(i * 2, 0) = fu * alpha(0);
        D(i * 2, 1) = fu * alpha(1);
        D(i * 2, 2) = fu * alpha(2);
        D(i * 2, 3) = fu * alpha(3);
        D(i * 2, 8) = (uc - p2d(0)) * alpha(0);
        D(i * 2, 9) = (uc - p2d(0)) * alpha(1);
        D(i * 2, 10) = (uc - p2d(0)) * alpha(2);
        D(i * 2, 11) = (uc - p2d(0)) * alpha(3);

        D(i * 2 + 1, 4) = fv * alpha(0);
        D(i * 2 + 1, 5) = fv * alpha(1);
        D(i * 2 + 1, 6) = fv * alpha(2);
        D(i * 2 + 1, 7) = fv * alpha(3);
        D(i * 2 + 1, 8) = (vc - p2d(1)) * alpha(0);
        D(i * 2 + 1, 9) = (vc - p2d(1)) * alpha(1);
        D(i * 2 + 1, 10) = (vc - p2d(1)) * alpha(2);
        D(i * 2 + 1, 11) = (vc - p2d(1)) * alpha(3);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullV | Eigen::ComputeFullU);
    auto V_matrix = svd.matrixV();
    cout << V_matrix << endl;
    auto V_vector = V_matrix.col(V_matrix.cols() - 1);
    cout << V_vector << endl;

    std::vector<Eigen::Vector3d> control_points_c(4);
    for (int i = 0; i < 4; i++) {
        double x = V_vector(i);
        double y = V_vector(i + 4);
        double z = V_vector(i + 8);
        Eigen::Vector3d p3d(x, y, z);
        control_points_c[i] = p3d;
    }

    //验证
//    {
//        for (int k = 0; k < p2ds.size(); k++) {
//            Eigen::Vector4d alpha = alphas[k];
//            Eigen::Vector2d p2d = p2ds[k];
//            Eigen::Vector3d p3d = p3ds[k];
//
//            Eigen::Vector3d tmp = Eigen::Vector3d::Zero();
//            for (int j = 0; j < 4; j++) {
//                tmp += control_points_c[j] * alpha[j];
//            }
//            tmp =  K *tmp;
//            tmp(0) /= tmp(2);
//            tmp(1) /= tmp(2);
//            tmp(2) /= tmp(2);
//            cout << "p2d:" << endl << p2d << endl;
//            cout << "tmp:" << endl << tmp << endl;
//            int a = 10;
//        }
//    }

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(12, 12);
    Eigen::VectorXd C_c = Eigen::VectorXd::Zero(12);

    for (int i = 0; i < 4; i++) {
        C_c(i * 3) = control_points_c[i](0);
        C_c(i * 3 + 1) = control_points_c[i](1);
        C_c(i * 3 + 2) = control_points_c[i](2);
        Q(i * 3, 0) = control_points_w[i](0);
        Q(i * 3, 1) = control_points_w[i](1);
        Q(i * 3, 2) = control_points_w[i](2);
        Q(i * 3, 9) = 1;

        Q(i * 3 + 1, 3) = control_points_w[i](0);
        Q(i * 3 + 1, 4) = control_points_w[i](1);
        Q(i * 3 + 1, 5) = control_points_w[i](2);
        Q(i * 3 + 1, 10) = 1;

        Q(i * 3 + 2, 6) = control_points_w[i](0);
        Q(i * 3 + 2, 7) = control_points_w[i](1);
        Q(i * 3 + 2, 8) = control_points_w[i](2);
        Q(i * 3 + 2, 11) = 1;

    }

    Eigen::MatrixXd Q_inv = Q.inverse();
    Eigen::VectorXd ans = Eigen::VectorXd::Zero(12);

    ans = Q_inv * C_c;

    R(0, 0) = ans(0);
    R(0, 1) = ans(1);
    R(0, 2) = ans(2);
    R(1, 0) = ans(3);
    R(1, 1) = ans(4);
    R(1, 2) = ans(5);
    R(2, 0) = ans(6);
    R(2, 1) = ans(7);
    R(2, 2) = ans(8);


    T(0) = ans(9);
    T(1) = ans(10);
    T(2) = ans(11);

    //验证
//    for (int i = 0; i < 3; i++) {
//        Eigen::Vector3d pc = control_points_c[i];
//        Eigen::Vector3d pw = control_points_w[i];
//        Eigen::Vector3d tmp = R * pw + T;
//
//        cout << "pw:" << pw << endl;
//        cout << "pc:" << pc << endl;
//        cout << "tmp:" << tmp << endl;
//        int a = 10;
//    }
}

bool GenerateTestData(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Matrix3d &K, std::vector<Eigen::Vector3d> &pts_3d,
                      std::vector<Eigen::Vector2d> &pts_2d) {

    double x, y, z;
    //随机生成100个三维点
    for (int i = 0; i < 100; i++) {
        x = (rand() % 5000) / 133.0;
        y = (rand() % 5000) / 213.0;
        z = (rand() % 5000) / 456.0;
        Eigen::Vector3d p3d(x, y, z);
        pts_3d.push_back(p3d);
        Eigen::Vector3d tmp = K * (R * p3d + t);
        Eigen::Vector2d p2d;
        p2d(0) = tmp(0) / tmp(2);
        p2d(1) = tmp(1) / tmp(2);
        pts_2d.push_back(p2d);
    }
    return true;
}

int main() {

    Eigen::Matrix3d K;
    K << 100, 0, 160,
            0, 100, 120,
            0, 0, 1;

    double R_nums[9];
    double t_nums[3];
    for (int i = 0; i < 9; i++) {
        R_nums[i] = (rand() % 4000) / 250.0;
    }
    for (int i = 0; i < 3; i++) {
        t_nums[i] = (rand() % 4000) / 350.0;
    }
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    t << R_nums[0], R_nums[1], R_nums[2];
    R << R_nums[0], R_nums[1], R_nums[2],
            R_nums[3], R_nums[4], R_nums[5],
            R_nums[6], R_nums[7], R_nums[8];

    std::vector<Eigen::Vector3d> pts_3d;
    std::vector<Eigen::Vector2d> pts_2d;

    GenerateTestData(R, t, K, pts_3d, pts_2d);

    Eigen::Matrix3d R_estimate, K_estimate;
    Eigen::Vector3d t_estimate;

    ePnP(pts_3d, pts_2d, K, R_estimate, t_estimate);

    cout << "R:" << endl << R << endl;
    cout << "R_estimate:" << endl << R_estimate << endl;
    cout << "t:" << endl << t << endl;
    cout << "t_estimate:" << endl << t_estimate << endl;

    //验证
    {
        for (int i = 0; i < pts_3d.size(); i++) {
            Eigen::Vector3d p3d = pts_3d[i];
            Eigen::Vector2d p2d = pts_2d[i];
            Eigen::Vector3d tmp = K * (R * p3d + t);
            tmp(0) /= tmp(2);
            tmp(1) /= tmp(2);
            tmp(2) /= tmp(2);
            cout << "p2d:" << endl << p2d << endl;
            cout << "tmp:" << endl << tmp << endl;
            int a = 10;
        }
    }




    return 0;
}