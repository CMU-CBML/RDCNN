#include <iostream>
#include <vector>
#include <array>
#include "PDE.h"
#include <sstream>
#include <iomanip>
#include <random>
#include <ctime>
#include "cxxopts.hpp"

using namespace std;

void Commandline(int argc, char* argv[]);

int main(int argc, char *argv[])
{

	Commandline(argc, argv);
	return 0;
}

void Commandline(int argc, char *argv[])
{

	try
	{
		cxxopts::Options options(argv[0], "CMU Solid Software");
		options
			.positional_help("[optional args]")
			.show_positional_help();

		bool flag_quality = false;

		// Dataset generation settings
		int time_per_set = 100;			  // Number of time steps extracted from each simulation
		int set_per_geo = 500;			  // Number of parameter settings for each geometry
		int num_geo_set = 21;			  // Number of different geometries
		string path("./");				  // Working directory
		string dir_output("../data_bc/"); // Folder to store all dataset

		bool flag_spline = false;
		bool flag_analysis = false;

		string fn_in;
		string fn_out;

options.add_options("General Settings")
			("h,help", "Print help")
			("t,nt", "Number of time steps extracted from each simulation", cxxopts::value<int>(time_per_set))
			("s,ns", "Number of parameter settings for each geometry", cxxopts::value<int>(set_per_geo))
			("g,ng", "Number of different geometries", cxxopts::value<int>(num_geo_set))
			("o,output", "Output folder", cxxopts::value<std::string>(dir_output))
#ifdef CXXOPTS_USE_UNICODE
			("unicode", u8"A help option with non-ascii: ��. Here the size of the"
				" string should be correct")
#endif
			;
		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			//cout << options.help({ "MeshQualityImprovement", "SplineConstruction", "Analysis" }) << std::endl;
			cout << options.help({"General Settings", "Mesh Quality Improvement", "Spline Construction"}) << endl;
			exit(0);
		}



		string fn_para(path + dir_output + "dataset_DKtGeo.txt");
		ofstream fout_para;
		fout_para.open(fn_para.c_str(), ios::app);

		int nstep(1000);
		double tstep(0.01);
		vector<int> hole_set;

#pragma omp parallel for ordered
		for (int aa = 0; aa < num_geo_set * set_per_geo; aa++)
		{
			PDE diffuse_react;
			diffuse_react.ReadMesh2D(path + "mesh_21_21.vtk");

			int count = aa * time_per_set; //200k + 82700

			random_device rd;
			mt19937 mt(rd());
			uniform_real_distribution<double> dist(0.0, 10.0);

			stringstream ss_hole;
			string fname_hole(path + "holeset/holeset_");

			int tmp_geo = aa / set_per_geo;
			int para_set = aa;
			ss_hole << tmp_geo << ".txt";
			diffuse_react.ReadHoleSet(fname_hole + ss_hole.str());

			double bc_set[4];
			for (int i = 0; i < 4; ++i)
				bc_set[i] = dist(mt) / 10.0;
			double D_diff = dist(mt);
			double K_reaction = dist(mt);

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
	}
	catch (const cxxopts::OptionException &e)
	{
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}
}

// Original hard code
// int main(int argc, char *argv[])
// {

// 	// Dataset generation settings
// 	int time_per_set = 100;			  // Number of time steps extracted from each simulation
// 	int set_per_geo = 500;			  // Number of parameter settings for each geometry
// 	int num_geo_set = 21;			  // Number of different geometries
// 	string path("./");				  // Working directory
// 	string dir_output("../data_bc/"); // Folder to store all dataset

// 	string fn_para(path + dir_output + "dataset_DKtGeo.txt");
// 	ofstream fout_para;
// 	fout_para.open(fn_para.c_str(), ios::app);

// 	int nstep(1000);
// 	double tstep(0.01);
// 	vector<int> hole_set;

// #pragma omp parallel for ordered
// 	for (int aa = 0; aa < num_geo_set * set_per_geo; aa++)
// 	{
// 		PDE diffuse_react;
// 		diffuse_react.ReadMesh2D(path + "mesh_21_21.vtk");

// 		int count = 282700 + aa * time_per_set; //200k + 82700

// 		random_device rd;
// 		mt19937 mt(rd());
// 		uniform_real_distribution<double> dist(0.0, 10.0);

// 		stringstream ss_hole;
// 		string fname_hole(path + "holeset/holeset_");

// 		int tmp_geo = aa / set_per_geo;
// 		int para_set = aa;
// 		ss_hole << tmp_geo << ".txt";
// 		diffuse_react.ReadHoleSet(fname_hole + ss_hole.str());

// 		double bc_set[4];
// 		for (int i = 0; i < 4; ++i)
// 			bc_set[i] = dist(mt) / 10.0;
// 		double D_diff = dist(mt);
// 		double K_reaction = dist(mt);

// 		diffuse_react.InitializeProblem(D_diff, K_reaction, tstep, bc_set);
// 		stringstream ss_fn1;
// 		ss_fn1 << "geometry_" << tmp_geo << "_" << para_set;
// 		string fn_out4(path + dir_output + "input/" + ss_fn1.str());
// 		diffuse_react.OutputInputData_ML(fn_out4, 21, 21);

// 		for (int i = 0; i < nstep; i++)
// 		{
// 			/*if ((i != 0 && (i + 1) % 10 == 0))
// 			{
// 				cout << "Step: " << i << "\n";
// 			}*/

// 			if ((i != 0 && (i + 1) % 10 == 0))
// 			{
// 				stringstream ss_fn;
// 				ss_fn << "mesh_" << count;
// 				//string fn_para("");
// 				//string fn_out1(path + dir_output + "physical/" + ss_fn.str());
// 				string fn_out2(path + dir_output + "parametric/" + ss_fn.str());
// 				string fn_out3(path + dir_output + "output/" + ss_fn.str());

// 				stringstream ss;
// 				if (i != 0)
// 					ss << (i + 1) * tstep;
// 				else
// 					ss << i;

// 				//diffuse_react.VisualizeVTK(fn_out1 + ss.str());
// 				//diffuse_react.VisualizeVTKParametric(fn_out2 + ss.str());
// 				//diffuse_react.OutputSolution_ML(fn_out3 + ss.str(), 21, 21);

// 				//diffuse_react.VisualizeVTK(fn_out1);
// 				diffuse_react.VisualizeVTKParametric(fn_out2);
// 				diffuse_react.OutputSolution_ML(fn_out3, 21, 21);
// //cout << "Step " << i << " done!\n";
// //getchar();
// #pragma omp critical
// 				{
// 					fout_para << count << "\t" << D_diff << "\t" << K_reaction << "\t" << (i + 1) * tstep << "\t" << bc_set[0] << "\t" << bc_set[1] << "\t" << bc_set[2] << "\t" << bc_set[3] << "\t" << tmp_geo << "\t" << para_set << "\n";
// 				}

// 				count++;
// 			}
// 			diffuse_react.Run();
// 		}
// 		cout << "DK settings: " << aa << " Done!\n";
// 		/*	clock_t end = clock();
// 		fout_time << ss_fn.str() << "     " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;*/
// 	}
// 	//fout_time.close();
// 	fout_para.close();

// 	cout << "DONE!\n";
// 	getchar();
// 	return 0;
// }