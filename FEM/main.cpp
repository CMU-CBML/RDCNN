#include <iostream>
#include <vector>
#include <array>
#include "PDE.h"
#include <sstream>
#include <iomanip>
#include <random>
#include <ctime>

using namespace std;

int main()
{
	int nstep(1000);
	double tstep(0.01);
	vector<int> hole_set;

	// double D_array[5] = { 1.,3.,6.,8.,10. };
	// double K_array[5] = { 1.,3.,6.,8.,10. };
	// double bc_array[5] = { 0.0, 0.25, 0.67, 0.8, 1.0 };
	// double bc_set[4] = { 2. / 3.,0.,0.,0. };
	// double D_diff;
	// double K_reaction;

	//nstep=21;

	string path("./");
	string dir_output("../data_bc/");

	// /// File to store the simulation running time
	// string fn_time(path + dir_output + "time_DKt.txt");
	// ofstream fout_time;
	// fout_time.open(fn_time.c_str(), ios::app);

	/// File to store the parameter information for each data file
	string fn_para(path + dir_output + "dataset_DKtGeo.txt");
	ofstream fout_para;
	fout_para.open(fn_para.c_str(), ios::app);

	///Scalar feature D, K, t (random number)

	//int count = 82700;
	//int count = 282700; //200k + 82700
	//int set_per_geo = 2;
	//int num_geo_set = 2;

	//int count = 282700; //200k + 82700
	int time_per_set = 100;
	int set_per_geo = 500;
	int num_geo_set = 21;

#pragma omp parallel for ordered
	for (int aa = 0; aa < num_geo_set * set_per_geo; aa++)
	{
		PDE diffuse_react;
		diffuse_react.ReadMesh2D(path + "mesh_21_21.vtk");

		int count = 282700 + aa * time_per_set; //200k + 82700

		random_device rd;
		mt19937 mt(rd());
		uniform_real_distribution<double> dist(0.0, 10.0);

		stringstream ss_hole;
		string fname_hole(path + "holeset/holeset_");

		int tmp_geo = aa / set_per_geo;
		int para_set = aa;
		ss_hole << tmp_geo << ".txt";
		diffuse_react.ReadHoleSet(fname_hole + ss_hole.str());

		/*for (int i = 0; i < 4; ++i)
			bc_set[i] = dist(mt)/10.0;
		D_diff = 1.0;
		K_reaction = 1.0;*/

		double bc_set[4];
		for (int i = 0; i < 4; ++i)
			bc_set[i] = dist(mt) / 10.0;
		double D_diff = dist(mt);
		double K_reaction = dist(mt);
		// D_diff = 1.0;
		// K_reaction = 8.0;
		// bc_set[0] = 2. / 3.;
		// bc_set[1] = 0.0;
		// bc_set[2] = 0.0;
		// bc_set[3] = 0.0;

		//ss_fn << "mesh_" << bc_set[0] << "_" << bc_set[1] << "_" << bc_set[2] << "_" << bc_set[3] << "_";

		//clock_t start = clock();

		diffuse_react.InitializeProblem(D_diff, K_reaction, tstep, bc_set);
		stringstream ss_fn1;
		ss_fn1 << "geometry_" << tmp_geo << "_" << para_set;
		string fn_out4(path + dir_output + "input/" + ss_fn1.str());
		diffuse_react.OutputInputData_ML(fn_out4, 21, 21);

		for (int i = 0; i < nstep; i++)
		{
			/*if ((i != 0 && (i + 1) % 10 == 0))
			{
				cout << "Step: " << i << "\n";
			}*/

			if ((i != 0 && (i + 1) % 10 == 0))
			{
				stringstream ss_fn;
				ss_fn << "mesh_" << count;
				//string fn_para("");
				//string fn_out1(path + dir_output + "physical/" + ss_fn.str());
				string fn_out2(path + dir_output + "parametric/" + ss_fn.str());
				string fn_out3(path + dir_output + "output/" + ss_fn.str());

				stringstream ss;
				if (i != 0)
					ss << (i + 1) * tstep;
				else
					ss << i;

				//diffuse_react.VisualizeVTK(fn_out1 + ss.str());
				//diffuse_react.VisualizeVTKParametric(fn_out2 + ss.str());
				//diffuse_react.OutputSolution_ML(fn_out3 + ss.str(), 21, 21);

				//diffuse_react.VisualizeVTK(fn_out1);
				diffuse_react.VisualizeVTKParametric(fn_out2);
				diffuse_react.OutputSolution_ML(fn_out3, 21, 21);
//cout << "Step " << i << " done!\n";
//getchar();
#pragma omp critical
				{
					fout_para << count << "\t" << D_diff << "\t" << K_reaction << "\t" << (i + 1) * tstep << "\t" << bc_set[0] << "\t" << bc_set[1] << "\t" << bc_set[2] << "\t" << bc_set[3] << "\t" << tmp_geo << "\t" << para_set << "\n";
				}

				count++;
			}
			diffuse_react.Run();
		}
		cout << "DK settings: " << aa << " Done!\n";
		/*	clock_t end = clock();
		fout_time << ss_fn.str() << "     " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;*/
	}
	//fout_time.close();
	fout_para.close();

	cout << "DONE!\n";
	getchar();
	return 0;
}
