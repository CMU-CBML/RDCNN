#include "BasicDataStructure.h"


Edge::Edge()
{
	pt[0] = -1; pt[1] = -1;
}

bool Edge::operator==(const Edge& ed)
{
	return ((pt[0] == ed.pt[0] && pt[1] == ed.pt[1]) || (pt[0] == ed.pt[1] && pt[1] == ed.pt[0]));
}

Vertex2D::Vertex2D()
{
	coor[0] = 0.;	coor[1] = 0.;	coor[2] = 0.;
	coortmp[0] = 0.;	coortmp[1] = 0.;	coortmp[2] = 0.;
}

Element2D::Element2D()
{
	cnct.resize(4);
	IEN.resize(4);
	pts.resize(4);
	for (int i = 0; i < 4; i++)
	{
		cnct[i] = 0;
		IEN[i] = 0;
		pts[i][0] = 0.; pts[i][1] = 0.; pts[i][2] = 0.;
	}
}

Element3D::Element3D()
{
	IEN.resize(8);
	pts.resize(8);
	for (int i = 0; i < 8; i++)
	{
		IEN[i] = 0;
		pts[i][0] = 0.; pts[i][1] = 0.; pts[i][2] = 0.;
	}
}


void Raw2Vtk_hex(string fn)
{
	unsigned int npt, nel;
	vector<array<double, 3>> pts;
	vector<array<int, 8>> cnct;
	double tmp;
	string fn1(fn + ".raw");
	ifstream fin;
	fin.open(fn1);
	if (fin.is_open())
	{
		fin >> npt >> nel;
		pts.resize(npt);
		cnct.resize(nel);
		for (unsigned int i = 0; i < npt; i++)
		{
			fin >> pts[i][0] >> pts[i][1] >> pts[i][2] >> tmp;
			fin >> tmp;
		}
		for (unsigned int i = 0; i < nel; i++)
		{
			for (int j = 0; j < 8; j++) fin >> cnct[i][j];
			fin >> tmp;
		}
		fin.close();
	}
	else
	{
		cerr << "Cannot open " << fn1 << "!\n";
	}
	string fn2(fn + ".vtk");
	ofstream fout;
	fout.open(fn2);
	if (fout.is_open())
	{
		fout << "# vtk DataFile Version 2.0\nHex test\nASCII\nDATASET UNSTRUCTURED_GRID\n";
		fout << "POINTS " << pts.size() << " float\n";
		for (unsigned int i = 0; i<pts.size(); i++)
		{
			fout << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << "\n";
		}
		fout << "\nCELLS " << cnct.size() << " " << 9 * cnct.size() << '\n';
		for (unsigned int i = 0; i<cnct.size(); i++)
		{
			fout << "8 ";
			for (int j = 0; j<8; j++)
			{
				fout << cnct[i][j] << ' ';
			}
			fout << '\n';
		}
		fout << "\nCELL_TYPES " << cnct.size() << '\n';
		for (unsigned int i = 0; i<cnct.size(); i++)
		{
			fout << "12\n";
		}
		fout.close();
	}
	else
	{
		cerr << "Cannot open " << fn2 << "!\n";
	}
}

void Rawn2Vtk_hex(string fn)
{
	unsigned int npt, nel;
	vector<array<double, 3>> pts;
	vector<array<int, 8>> cnct;
	double tmp;
	string fn1(fn + ".rawn");
	ifstream fin;
	fin.open(fn1);
	if (fin.is_open())
	{
		fin >> npt >> nel;
		pts.resize(npt);
		cnct.resize(nel);
		for (unsigned int i = 0; i < npt; i++)
		{
			fin >> pts[i][0] >> pts[i][1] >> pts[i][2] >> tmp >> tmp >> tmp >> tmp;
		}
		for (unsigned int i = 0; i < nel; i++)
		{
			for (int j = 0; j < 8; j++) fin >> cnct[i][j];
		}
		fin.close();
	}
	else
	{
		cerr << "Cannot open " << fn1 << "!\n";
	}
	string fn2(fn + ".vtk");
	ofstream fout;
	fout.open(fn2);
	if (fout.is_open())
	{
		fout << "# vtk DataFile Version 2.0\nHex test\nASCII\nDATASET UNSTRUCTURED_GRID\n";
		fout << "POINTS " << pts.size() << " float\n";
		for (unsigned int i = 0; i<pts.size(); i++)
		{
			fout << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << "\n";
		}
		fout << "\nCELLS " << cnct.size() << " " << 9 * cnct.size() << '\n';
		for (unsigned int i = 0; i<cnct.size(); i++)
		{
			fout << "8 ";
			for (int j = 0; j<8; j++)
			{
				fout << cnct[i][j] << ' ';
			}
			fout << '\n';
		}
		fout << "\nCELL_TYPES " << cnct.size() << '\n';
		for (unsigned int i = 0; i<cnct.size(); i++)
		{
			fout << "12\n";
		}
		fout.close();
	}
	else
	{
		cerr << "Cannot open " << fn2 << "!\n";
	}
}

