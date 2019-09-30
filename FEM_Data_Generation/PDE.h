#ifndef REACTION_H
#define REACTION_H

#include <vector>
#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "BasicDataStructure.h"

using namespace std;
using namespace Eigen;

class PDE
{
public:
	PDE();
	void GaussInfo(int ng=2);
	void ReadMesh2D(string fn);
	void InitializeProblem(double D_diff, double K_reaction, double tstep, double Ubc[4]);
    void ParametricMapping();
	void BuildInitialEdges();
	void ReadHoleSet(string filename);

	void BasisFunction(double u, double v, const vector<array<double,3>>& pt, double Nx[4], double dNdx[4][2], double& detJ);
	void ElementValue(double Nx[4], double value_node[4], double &result);
	void ElementDerivative(double dNdx[4][2], double value_node[4], double result[2]);
	void ElementMassMatrix(double Nx[4], double detJ, double EM[4][4]);
	void ElementStiffMatrix(double dNdx[4][2], double detJ, double EK[4][4]);
	void ElementReactionMatrix(double u_ele, double Nx[4], double detJ, double ER[4][4]);
	void ElementLoadVector(double u_ele, double Nx[4], double detJ, double EF[4]);
	void ElementAssembly(double u_node[4], double EM[4][4], double EK[4][4],double ER[4][4], double EF[4], double EMatrixSolve[4][4], double EVecotrSolve[4]);
	void ApplyBoundaryCondition(int index, double bc_value, double MatrixSolve[4][4], double VectorSolve[4]);

	void Residual(double u_t[4], double u_node[4], double Nx[4], double dNdx[4][2], double detJ, double Re[4]);
	void Tangent(double u_t[4], double u_node[4], double Nx[4], double dNdx[4][2], double detJ, double Ke[4][4]);

	void GlobalAssembly(double EMatrixSolve[4][4], double EVecotrSolve[4], const vector<int>& IEN, SparseMatrix<double>& GMatrixSolve, vector<double> &GVectorSolve);
	void BuildLinearSystem(vector<double>& Umid, vector<double>& UTmid, vector<double>& U0, vector<double>& UT0, SparseMatrix<double>& GMatrixSolve, vector<double>& GVectorSolve);
	void Solver(SparseMatrix<double>& GM, SparseMatrix<double>& GK, SparseMatrix<double>& GM_CA, vector<double>& Bv);
	void VisualizeVTK(string fn);
	void VisualizeVTKParametric(string fn);
	void OutputInputData_ML(string fn, int nrow, int ncol);
	void OutputSolution_ML(string fn, int nrow, int ncol);
	void Run();
	
private:
	vector<double> Gpt;
	vector<double> wght;

	vector<Element2D> tmesh;
	vector<Edge> tmedge;
	vector<int> IDBC;
	vector<int> holeset;
	vector<Vertex2D> pts;
	vector<Vertex2D> pts_parametric;
	vector<int> pid_loc;

	double D; //diffusion coefficient
	double kr; //reaction coefficient
	double u_bc[4] = {2./3., 0., 0., 0.}; //left, inner, bottom, outer
	double alphaM, alphaF, rou, gama;
	double dt;
	vector<double> uh;
	vector<double> uh_t;
};

#endif