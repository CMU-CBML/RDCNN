#ifndef BASIC_DATA_STRUCTURE_H
#define BASIC_DATA_STRUCTURE_H

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip> 

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

//const int phys_dim(3);

/////////////////////////////////

class Vertex2D
{
public:
	double coor[3];
	double coortmp[3];
	vector<int> edge;//edges that connect to this vertex
	vector<int> face;//faces that share this vertex
	Vertex2D();
	//bool operator==(const Vertex2D& v);
};

class Edge
{
public:
	int pt[2];
	vector<int> face;
	bool operator==(const Edge& ed);

	Edge();
};

class Element2D
{
public:
	vector<int> cnct;
	vector<int> IEN;
	vector<array<double, 3>> pts;//tmp
	int edge[4];
	Element2D();
};

class Element3D
{
public:
	vector<int> IEN;
	vector<array<double, 3>> pts;//tmp
	int edge[12];

	Element3D();
};


//mesh format conversion

void Raw2Vtk_hex(string fn);

void Rawn2Vtk_hex(string fn);

#endif