#include "PDE.h"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

typedef unsigned int uint;

PDE::PDE()
{
}

void PDE::GaussInfo(int ng)
{
	Gpt.clear();
	wght.clear();
	switch(ng)
	{
	case 2:
		{
			Gpt.resize(ng);
			wght.resize(ng);
			Gpt[0]=0.2113248654051871;			Gpt[1]=0.7886751345948129;
			wght[0]=1.;			wght[1]=1.;
			break;
		}
	case 3:
		{
			Gpt.resize(ng);
			wght.resize(ng);
			Gpt[0]=0.1127016653792583;			Gpt[1]=0.5;			Gpt[2]=0.8872983346207417;
			wght[0]=0.5555555555555556;			wght[1]=0.8888888888888889;			wght[2]=0.5555555555555556;
			break;
		}
	case 4:
		{
			Gpt.resize(ng);
			wght.resize(ng);
			Gpt[0]=0.06943184420297371;			Gpt[1]=0.33000947820757187;			Gpt[2]=0.6699905217924281;			Gpt[3]=0.9305681557970262;
			wght[0]=0.3478548451374539;			wght[1]=0.6521451548625461;			wght[2]=0.6521451548625461;			wght[3]=0.3478548451374539;
			break;
		}
	default:
		{
			Gpt.resize(2);
			wght.resize(2);
			Gpt[0]=0.2113248654051871;			Gpt[1]=0.7886751345948129;
			wght[0]=1.;			wght[1]=1.;
			break;
		}
	}
}

void PDE::InitializeProblem(double D_diff, double K_reaction, double tstep, double Ubc[4])
{
	uh.clear();
	uh_t.clear();
	uh.resize(pts.size(), 0.1);
	uh_t.resize(pts.size(), 0.0);
	D = D_diff;  //diffusion coefficient
	kr = K_reaction; //reaction coefficient
	dt = tstep;
	for (int i = 0; i < 4; i++) 
		u_bc[i] = Ubc[i]; //Initialize Dirichlet BC

	rou = 0.5;
	alphaF = 1.0 / (1.0 + rou);
	alphaM = 0.5 * (3.0 - rou) / (1.0 + rou);
	gama = 0.5 + alphaM - alphaF;
	ParametricMapping();
}

void PDE::ReadHoleSet(string filename)
{
	holeset.clear();
	ifstream fin;
	int tmp;
	fin.open(filename);
	while (!fin.eof())
	{
		fin >> tmp;
		holeset.push_back(tmp);
	}
	/*cout << holeset.size() << endl;
	for (int i = 0; i < holeset.size(); i++)
		cout << holeset[i] << endl;*/
	cout << "Finish Reading"<< filename << endl;
}

void PDE::BuildInitialEdges()
{
	int edloc[4][2] = { {0,1},{1,2},{2,3},{3,0} };
	uint i, j, k;
	tmedge.clear();

	//point-quad relation
	for (i = 0; i < tmesh.size(); i++)
	{
		
		for (i = 0; i < tmesh.size(); i++)
		{
			for (j = 0; j < 4; j++)
			{
				pts[tmesh[i].cnct[j]].face.push_back(i);
			}
		}
	}

	//construct edge
	for (i = 0; i < tmesh.size(); i++)
	{
		vector<int> nb;
		for (j = 0; j < 4; j++)
		{
			for (k = 0; k < pts[tmesh[i].cnct[j]].face.size(); k++)
			{
				int eid(pts[tmesh[i].cnct[j]].face[k]);
				if (eid < i)
				{
					vector<int>::iterator it = find(nb.begin(), nb.end(), eid);
					if (it == nb.end())
					{
						nb.push_back(eid);
					}
				}
			}
		}		
		for (j = 0; j < 4; j++) //edge
		{
			Edge edtmp;
			edtmp.pt[0] = tmesh[i].cnct[edloc[j][0]];
			edtmp.pt[1] = tmesh[i].cnct[edloc[j][1]];
			int flag(-1);
			for (k = 0; k < nb.size(); k++)
			{
				for (int k0 = 0; k0 < 4; k0++)
				{
					if (edtmp == tmedge[tmesh[nb[k]].edge[k0]])
					{
						flag = tmesh[nb[k]].edge[k0];
						break;
					}
				}
				if (flag != -1)
					break;
			}
			if (flag != -1)
			{
				tmesh[i].edge[j] = flag;
			}
			else
			{
				tmedge.push_back(edtmp);
				tmesh[i].edge[j] = tmedge.size() - 1;
			}
		}
	}
	for (i = 0; i<tmesh.size(); i++)
	{
		for (j = 0; j<4; j++)
		{
			tmedge[tmesh[i].edge[j]].face.push_back(i);
		}
	}
	for (i = 0; i<tmedge.size(); i++)
	{
		for (j = 0; j<2; j++)
		{
			pts[tmedge[i].pt[j]].edge.push_back(i);
		}
	}
}

void PDE::ParametricMapping()
{
	pts_parametric.resize(pts.size());
	double r, theta;
	for (int i = 0; i < pts_parametric.size(); i++)
	{
		r = sqrt(pts[i].coor[0] * pts[i].coor[0] + pts[i].coor[1] * pts[i].coor[1]);
		theta = acos(pts[i].coor[0] / r);
		pts_parametric[i].coor[0] = r / 5.0;
		pts_parametric[i].coor[1] = theta / atan(1.0) / 2.0;
	}
}

void PDE::ReadMesh2D(string fn) //need vtk file with point label
{
	string fname(fn), stmp;
	int npts, neles, itmp;
	ifstream fin;
	fin.open(fname);
	if (fin.is_open())
	{
		for (int i = 0; i < 4; i++)
			getline(fin, stmp); //skip lines
		fin >> stmp >> npts >> stmp;
		pts.resize(npts);
		IDBC.resize(npts);
		for (int i = 0; i < npts; i++)
		{
			fin >> pts[i].coor[0] >> pts[i].coor[1] >> itmp;
		}
		getline(fin, stmp);
		fin >> stmp >> neles >> itmp;
		tmesh.resize(neles);
		for (int i = 0; i < neles; i++)
		{
			fin >> itmp >> tmesh[i].IEN[0] >> tmesh[i].IEN[1] >> tmesh[i].IEN[2] >> tmesh[i].IEN[3];
			for (int j = 0; j < 4; j++)
			{
				tmesh[i].cnct[j] = tmesh[i].IEN[j];
				tmesh[i].pts[j][0] = pts[tmesh[i].IEN[j]].coor[0];
				tmesh[i].pts[j][1] = pts[tmesh[i].IEN[j]].coor[1];
			}
		}
		//cout << "Elements Number: " << tmesh.size() << endl;
		BuildInitialEdges();
		//cout << "Edge Number: "<<tmedge.size()<<endl;

		fin.close();

		////import IDBC information
		//getline(fin, stmp);
		//fin >> stmp >> neles;
		//for (int i = 0; i<neles; i++)
		//	fin >> itmp;
		//getline(fin, stmp);
		//for (int i = 0; i < 4; i++)
		//	getline(fin, stmp);
		//
		//for (int i = 0; i < npts; i++)
		//	fin >> IDBC[i];
		//fin.close();
		cout << "Mesh Loaded!" << endl;
	}
	else
	{
		cerr << "Cannot open " << fname << "!\n";
	}
}

void PDE::BasisFunction(double u, double v, const vector<array<double, 3>> &pt, double Nx[4], double dNdx[4][2], double &detJ)
{
	double Nu[2] = {(1. - u), u};
	double Nv[2] = {(1. - v), v};
	double dNdu[2] = {-1., 1.};
	double dNdv[2] = {-1., 1.};
	double dNdt[4][2];
	double X(0), Y(0);

	int i, j, k, a, b, loc;
	loc = 0;

	for (j = 0; j < 2; j++)
	{
		X += pt[loc][0] * Nu[j] * Nv[0];
		Y += pt[loc][1] * Nu[j] * Nv[0];
		loc++;
	}
	for (j = 1; j >= 0; j--)
	{
		X += pt[loc][0] * Nu[j] * Nv[1];
		Y += pt[loc][1] * Nu[j] * Nv[1];
		loc++;
	}

	loc = 0;
	for (k = 0; k < 2; k++)
	{
		Nx[loc] = Nu[k] * Nv[0];
		dNdt[loc][0] = dNdu[k] * Nv[0];
		dNdt[loc][1] = Nu[k] * dNdv[0];
		loc++;
	}
	for (k = 1; k >= 0; k--)
	{
		Nx[loc] = Nu[k] * Nv[1];
		dNdt[loc][0] = dNdu[k] * Nv[1];
		dNdt[loc][1] = Nu[k] * dNdv[1];
		loc++;
	}

	Matrix2d dxdt = Matrix2d::Zero();
	loc = 0;
	for (i = 0; i < 2; i++)
	{
		for (j = 0; j < 2; j++)
		{
			for (a = 0; a < 2; a++)
			{
				for (b = 0; b < 2; b++)
				{
					dxdt(a, b) += pt[loc][a] * dNdt[loc][b];
				}
			}
			loc++;
		}
	}

	Matrix2d dtdx = dxdt.inverse();
	for (i = 0; i < 4; i++)
	{
		dNdx[i][0] = dNdt[i][0] * dtdx(0, 0) + dNdt[i][1] * dtdx(1, 0);
		dNdx[i][1] = dNdt[i][0] * dtdx(0, 1) + dNdt[i][1] * dtdx(1, 1);
	}
	detJ = dxdt.determinant();
	detJ = 0.25 * detJ;
}

void PDE::ElementValue(double Nx[4], double value_node[4], double &result)
{
	result = 0.;
	for (int i = 0; i < 4; i++)
	{
		result += value_node[i] * Nx[i];
	}
}

void PDE::ElementDerivative(double dNdx[4][2], double value_node[4], double result[2])
{
	result[0] = 0.;
	result[1] = 0.;
	for (int i = 0; i < 4; i++)
	{
		result[0] += value_node[i] * dNdx[i][0];
		result[1] += value_node[i] * dNdx[i][1];
	}
}

void PDE::ElementMassMatrix(double Nx[4], double detJ, double EM[4][4])
{
	int i, j;
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			EM[i][j] += Nx[i] * Nx[j] * detJ;
		}
	}
}

void PDE::ElementStiffMatrix(double dNdx[4][2], double detJ, double EK[4][4])
{
	int i, j;
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			EK[i][j] += (dNdx[i][0] * dNdx[j][0] + dNdx[i][1] * dNdx[j][1]) * detJ;
		}
	}
}

void PDE::ElementReactionMatrix(double u_ele, double Nx[4], double detJ, double ER[4][4])
{
	int i, j;
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			ER[i][j] += kr * u_ele * (1.0 - u_ele) * Nx[i] * Nx[j] * detJ;
		}
	}
}

void PDE::ElementLoadVector(double u_ele, double Nx[4], double detJ, double EF[4])
{
	int i;
	double u2 = u_ele * u_ele;
	double u3 = u_ele * u_ele * u_ele;
	for (i = 0; i < 4; i++)
	{
		//unsteady test
		EF[i] += Nx[i] * kr * (u2 - u3) * detJ;
		//EF[i] = 0.;
		//steady test
		//EF[i] += Nx[i] * detJ;
	}
}

void PDE::ElementAssembly(double u_node[4], double EM[4][4], double EK[4][4], double ER[4][4], double EF[4], double EMatrixSolve[4][4], double EVecotrSolve[4])
{
	int i, j;
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			////unsteady
			//EMatrixSolve[i][j] += EM[i][j] + dt*D*EK[i][j]-dt*ER[i][j];
			EMatrixSolve[i][j] += EM[i][j] + dt * D * EK[i][j];
			EVecotrSolve[i] += u_node[j] * (EM[i][j]);
			////steady
			//EMatrixSolve[i][j] += EK[i][j];
			//Newton's method
		}
		////unsteady
		EVecotrSolve[i] += dt * EF[i];
		////steady
		//EVecotrSolve[i] += EF[i];
	}
}

void PDE::Residual(double u_t[4], double u_node[4], double Nx[4], double dNdx[4][2], double detJ, double Re[4])
{
	int i;
	double u_ele, ut_ele;
	double dudx_ele[2];
	ElementValue(Nx, u_node, u_ele);
	ElementValue(Nx, u_t, ut_ele);
	ElementDerivative(dNdx, u_node, dudx_ele);
	for (i = 0; i < 4; i++)
	{
		Re[i] -= (ut_ele * Nx[i] + D * (dNdx[i][0] * dudx_ele[0] + dNdx[i][1] * dudx_ele[1]) - Nx[i] * kr * u_ele * u_ele * (1 - u_ele)) * detJ;
	}
}

void PDE::Tangent(double u_t[4], double u_node[4], double Nx[4], double dNdx[4][2], double detJ, double Ke[4][4])
{
	int i, j;
	double u_ele, ut_ele;
	double dudx_ele[2];
	ElementValue(Nx, u_node, u_ele);
	ElementValue(Nx, u_t, ut_ele);
	ElementDerivative(dNdx, u_node, dudx_ele);
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			Ke[i][j] += (alphaM * Nx[i] * Nx[j] + alphaF * gama * dt * (D * (dNdx[i][0] * dNdx[j][0] + dNdx[i][1] * dNdx[j][1]) - Nx[i] * kr * (2 * Nx[j] - 3 * u_ele * Nx[j]))) * detJ;
		}
	}
}

void PDE::ApplyBoundaryCondition(int index, double bc_value, double MatrixSolve[4][4], double VectorSolve[4])
{
	int i;
	for (i = 0; i < 4; i++)
	{
		if (i == index)
		{
			VectorSolve[i] -= bc_value * MatrixSolve[i][index];
		}
	}
	for (i = 0; i < 4; i++)
	{
		MatrixSolve[i][index] = 0.0;
		MatrixSolve[index][i] = 0.0;
	}
	MatrixSolve[index][index] = 1.0;
	VectorSolve[index] = bc_value;
}

void PDE::GlobalAssembly(double EMatrixSolve[4][4], double EVecotrSolve[4], const vector<int> &IEN, SparseMatrix<double> &GMatrixSolve, vector<double> &GVectorSolve)
{
	unsigned int i, j, A, B;
	for (i = 0; i < IEN.size(); i++)
	{
		A = IEN[i];
		for (j = 0; j < IEN.size(); j++)
		{
			B = IEN[j];
			GMatrixSolve.coeffRef(A, B) += EMatrixSolve[i][j];
		}
		GVectorSolve[A] += EVecotrSolve[i];
	}
}

void PDE::BuildLinearSystem(vector<double> &Umid, vector<double> &UTmid, vector<double> &U0, vector<double> &UT0, SparseMatrix<double> &GMatrixSolve, vector<double> &GVectorSolve)
{
	unsigned int e, i, j, k, a, b;
	double EM[4][4], EK[4][4], ER[4][4], EF[4];
	double Re[4], Ke[4][4];
	double EMatrixSolve[4][4], EVectorSolve[4];
	double Nx[4];
	double dNdx[4][2];
	double detJ;
	double u_ele, ut_node[4], u_node[4];
	for (e = 0; e < tmesh.size(); e++)
	{
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 4; j++)
			{
				EM[i][j] = 0.;
				EK[i][j] = 0.;
				ER[i][j] = 0.;
				Ke[i][j] = 0.;
				EMatrixSolve[i][j] = 0.;
			}
			EF[i] = 0.;
			Re[i] = 0.;
			EVectorSolve[i] = 0.;
		}
		for (i = 0; i < 4; i++)
		{
			u_node[i] = Umid[tmesh[e].IEN[i]];
			ut_node[i] = UTmid[tmesh[e].IEN[i]];
		}
		for (i = 0; i < Gpt.size(); i++)
		{
			for (j = 0; j < Gpt.size(); j++)
			{
				BasisFunction(Gpt[i], Gpt[j], tmesh[e].pts, Nx, dNdx, detJ);
				detJ = wght[i] * wght[j] * detJ;
				Tangent(ut_node, u_node, Nx, dNdx, detJ, EMatrixSolve);
				Residual(ut_node, u_node, Nx, dNdx, detJ, EVectorSolve);
			}
		}

		//ElementAssembly(u_node, EM, EK, ER, EF, EMatrixSolve, EVectorSolve);
		//ApplyBoundaryCondition
		for (i = 0; i < 4; i++)
		{
			//int A = IDBC[tmesh[e].IEN[i]];
			int A = tmesh[e].IEN[i];
			//Apply BCs using geometry
			double R = sqrt(tmesh[e].pts[i][0] * tmesh[e].pts[i][0] + tmesh[e].pts[i][1] * tmesh[e].pts[i][1]);
			double bc;
			if (R < 5.1) //wall
			{
				bc = (u_bc[1] - U0[A]) / gama / dt;
				ApplyBoundaryCondition(i, bc, EMatrixSolve, EVectorSolve);
			}
			else if (R > 9.9)
			{
				bc = (u_bc[3] - U0[A]) / gama / dt;
				ApplyBoundaryCondition(i, bc, EMatrixSolve, EVectorSolve);
			}
			else if (tmesh[e].pts[i][0] < 0.1) //inlet
			{
				bc = (u_bc[0] - U0[A]) / gama / dt;
				ApplyBoundaryCondition(i, bc, EMatrixSolve, EVectorSolve);
			}
			else if (tmesh[e].pts[i][1] < 0.1) //outlet
			{
				bc = (u_bc[2] - U0[A]) / gama / dt;
				ApplyBoundaryCondition(i, bc, EMatrixSolve, EVectorSolve);
			}

			///Apply BCs using holewholeset
			double bc_value = 0.0;
			if (find(holeset.begin(), holeset.end(), A) != holeset.end())
			{
				bc_value = (0.0 - U0[A]) / gama / dt;
				ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			}

			///Apply BCs using node list
			//double bc_value;
			//vector<int> bc1{ 5, 11, 17, 23, 29, 176, 177, 178, 179}; //South
			//vector<int> bc2{ 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65};//SouthWest
			//vector<int> bc3{ 36, 41, 46, 51, 56, 66, 67, 68, 69};//SouthEast
			//vector<int> bc4{ 70, 75, 80, 85, 90, 95, 101, 107, 113, 119, 125};//East
			//vector<int> bc5{ 120, 121, 122, 123, 124, 131, 137, 143, 149};//NorthEast
			//vector<int> bc6{ 150, 151, 152, 153, 154, 155, 160, 165, 170, 175, 180};//NorthWest
			//if (find(bc1.begin(), bc1.end(), A) != bc1.end())
			//{
			//	bc_value = (2./3. - U0[A]) / gama / dt;
			//	ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			//}
			//if (find(bc2.begin(), bc2.end(), A) != bc2.end())
			//{
			//	bc_value = (0.0 - U0[A]) / gama / dt;
			//	ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			//}
			//if (find(bc3.begin(), bc3.end(), A) != bc3.end())
			//{
			//	bc_value = (0.0 - U0[A]) / gama / dt;
			//	ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			//}
			//if (find(bc4.begin(), bc4.end(), A) != bc4.end())
			//{
			//	bc_value = (0.0 - U0[A]) / gama / dt;
			//	ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			//}
			//if (find(bc5.begin(), bc5.end(), A) != bc5.end())
			//{
			//	bc_value = (0.0 - U0[A]) / gama / dt;
			//	ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			//}
			//if (find(bc6.begin(), bc6.end(), A) != bc6.end())
			//{
			//	bc_value = (0.0 - U0[A]) / gama / dt;
			//	ApplyBoundaryCondition(i, bc_value, EMatrixSolve, EVectorSolve);
			//}

		}

		GlobalAssembly(EMatrixSolve, EVectorSolve, tmesh[e].IEN, GMatrixSolve, GVectorSolve);
	}
}

void PDE::VisualizeVTK(string fn)
{
	string fname = fn + "_uh.vtk";
	ofstream fout;
	fout.open(fname.c_str());
	unsigned int i;
	if (fout.is_open())
	{
		fout << "# vtk DataFile Version 2.0\nSquare plate test\nASCII\nDATASET UNSTRUCTURED_GRID\n";
		fout << "POINTS " << pts.size() << " float\n";
		for (i = 0; i < pts.size(); i++)
		{
			fout << pts[i].coor[0] << " " << pts[i].coor[1] << " " << 0 << "\n";
		}
		fout << "\nCELLS " << tmesh.size() << " " << 5 * tmesh.size() << '\n';
		for (i = 0; i < tmesh.size(); i++)
		{
			fout << "4 " << tmesh[i].IEN[0] << " " << tmesh[i].IEN[1] << " " << tmesh[i].IEN[2] << " " << tmesh[i].IEN[3] << '\n';
		}
		fout << "\nCELL_TYPES " << tmesh.size() << '\n';
		for (i = 0; i < tmesh.size(); i++)
		{
			fout << "9\n";
		}
		fout << "POINT_DATA " << uh.size() << "\nSCALARS uh float 1\nLOOKUP_TABLE default\n";
		for (uint i = 0; i < uh.size(); i++)
		{
			fout << uh[i] << "\n";
		}
		fout.close();
	}
	else
	{
		cout << "Cannot open " << fname << "!\n";
	}
}

void PDE::VisualizeVTKParametric(string fn)
{
	string fname = fn + "_uh.vtk";
	ofstream fout;
	fout.open(fname.c_str());
	unsigned int i;
	if (fout.is_open())
	{
		fout << "# vtk DataFile Version 2.0\nSquare plate test\nASCII\nDATASET UNSTRUCTURED_GRID\n";
		fout << "POINTS " << pts_parametric.size() << " float\n";
		for (i = 0; i < pts_parametric.size(); i++)
		{
			fout << pts_parametric[i].coor[0] << " " << pts_parametric[i].coor[1] << " " << 0 << "\n";
		}
		fout << "\nCELLS " << tmesh.size() << " " << 5 * tmesh.size() << '\n';
		for (i = 0; i < tmesh.size(); i++)
		{
			fout << "4 " << tmesh[i].IEN[0] << " " << tmesh[i].IEN[1] << " " << tmesh[i].IEN[2] << " " << tmesh[i].IEN[3] << '\n';
		}
		fout << "\nCELL_TYPES " << tmesh.size() << '\n';
		for (i = 0; i < tmesh.size(); i++)
		{
			fout << "9\n";
		}
		fout << "POINT_DATA " << uh.size() << "\nSCALARS uh float 1\nLOOKUP_TABLE default\n";
		for (uint i = 0; i < uh.size(); i++)
		{
			fout << uh[i] << "\n";
		}
		fout.close();
	}
	else
	{
		cout << "Cannot open " << fname << "!\n";
	}
}

void PDE::OutputInputData_ML(string fn, int nrow, int ncol)
{

	string fname = fn + "_input.txt";
	ofstream fout;
	fout.open(fname.c_str());
	int n;
	if (fout.is_open())
	{
		for (int i = 0; i < nrow; i++)
		{
			for (int j = 0; j < ncol; j++)
			{
				int A = i * ncol + j;
				
				double R = sqrt(pts[A].coor[0] * pts[A].coor[0] + pts[A].coor[1] * pts[A].coor[1]);
				double ini_val = -1;
				if (R < 5.1) //wall
					ini_val = u_bc[1];
				else if (R > 9.9)
					ini_val = u_bc[3];
				else if (pts[A].coor[0] < 0.1) //inlet
					ini_val = u_bc[0];
				else if (pts[A].coor[1] < 0.1) //outlet
					ini_val = u_bc[2];

				if (find(holeset.begin(), holeset.end(), A) != holeset.end())
				{
					ini_val = -2;
				}

				fout << i << " " << j << " " << ini_val << "\n";
			}
		}
		fout.close();
	}
	else
	{
		cout << "Cannot open " << fname << "!\n";
	}

}

void PDE::OutputSolution_ML(string fn, int nrow, int ncol)
{
	
	string fname = fn + ".txt";
	ofstream fout;
	fout.open(fname.c_str());
	int n;
	if (fout.is_open())
	{
		for (int i = 0; i < nrow; i++)
		{
			for (int j = 0; j < ncol; j++)
			{
				fout << i << " " << j << " " << uh[i * ncol + j] << "\n";
			}
		}
		fout.close();
	}
	else
	{
		cout << "Cannot open " << fname << "!\n";
	}
	
}

void PDE::Run()
{
	//cout << "Reaction...\n";
	GaussInfo(3);
	int n_iterator(3);
	vector<double> U0(pts.size()), UT0(pts.size());
	vector<double> Umid(pts.size()), UTmid(pts.size());
	vector<double> U_delta(pts.size()), UT_delta(pts.size());
	VectorXd tmp_solution(pts.size()), tmp_GF(pts.size());
	//cout << "Building linear system...\n";

	SparseMatrix<double> GMatrixSolve(uh.size(), uh.size());
	vector<double> GVectorSolve(uh.size(), 0.0);
	GMatrixSolve.setZero();

	//BuildLinearSystem(GMatrixSolve, GVectorSolve);

	//VectorXd tmp(uh.size());
	//for (uint i = 0; i < uh.size(); i++) tmp[i] = GVectorSolve[i];
	//
	SparseLU<SparseMatrix<double>> solver;
	//solver.compute(GMatrixSolve);
	//VectorXd sol = solver.solve(tmp);
	//for (uint i = 0; i<uh.size(); i++)
	//{
	//	uh[i] = sol[i];
	//}

	//cout << "Iterating...\n";

	for (uint i = 0; i < pts.size(); i++)
	{
		U0[i] = uh[i];
		UT0[i] = (gama - 1) / gama * uh_t[i];
	}

	for (uint i = 0; i < pts.size(); i++)
	{
		Umid[i] = uh[i] + (U0[i] - uh[i]) * alphaF;
		UTmid[i] = uh_t[i] + (UT0[i] - uh_t[i]) * alphaM;
	}

	for (int l = 0; l < n_iterator; l++)
	{
		//cout << "Iteration Step:" << l << "\n";
		//cout << "Building Linear System...\n";

		GMatrixSolve.setZero();
		GVectorSolve.clear();
		GVectorSolve.resize(pts.size(), 0.0);
		BuildLinearSystem(Umid, UTmid, U0, UT0, GMatrixSolve, GVectorSolve);

		for (uint i = 0; i < GVectorSolve.size(); i++)
		{
			tmp_GF[i] = GVectorSolve[i];
		}

		//cout << "Solving...\n";
		solver.compute(GMatrixSolve);
		tmp_solution = solver.solve(tmp_GF);

		for (uint i = 0; i < pts.size(); i++)
		{
			UT_delta[i] = tmp_solution[i];
		}

		for (uint i = 0; i < pts.size(); i++)
		{
			UT0[i] += UT_delta[i];
			U0[i] += gama * dt * UT_delta[i];
		}

		for (uint i = 0; i < pts.size(); i++)
		{
			Umid[i] = uh[i] + (U0[i] - uh[i]) * alphaF;
			UTmid[i] = uh_t[i] + (UT0[i] - uh_t[i]) * alphaM;
		}
	}

	for (uint i = 0; i < pts.size(); i++)
	{
		uh[i] = U0[i];
		uh_t[i] = UT0[i];
	}
	//for (uint i = 0; i < pts.size(); i++)
	//{
	//	Vel[2 * i] = V0[2 * i];
	//	Vel[2 * i + 1] = V0[2 * i + 1];
	//	VelDot[2 * i] = Vdot0[2 * i];
	//	VelDot[2 * i + 1] = Vdot0[2 * i + 1];
	//	Pre[i] = P0[i];
	//}

	//cout << "Done reaction!\n";
}