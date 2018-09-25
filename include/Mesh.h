#pragma once
#include<vector>
#include<string>
#include<Eigen/Dense>
#include "Element.h"
#include<petscksp.h>

class Mesh {
public:
	Mesh(std::string&); // constructor
	void readMesh(); // read mesh-data from file
	const std::vector<Element>& getGrid() const; // return reference to grid
	void Assemble(); // assemble linear system
	void Solve(); // solve linear system for unknowns
        void writeField(const std::string&); // write solution-field to file
        void computeL2Error(); // calculate L2 error
	~Mesh(); // destructor

private:  
        std::string meshFileName; // input mesh-filename
	std::vector <std::vector<double>> nodes; // node-list for grid
	std::vector <std::vector<int>> elements; // connectivities
	std::vector<Element> mesh; // grid-vector containing list of elements
	//std::vector<std::vector<double>> globalStiffness; // global stiffness matrix for grid
	//std::vector<double> globalForce; // global force vector
    Eigen::VectorXd x_sol; // solution-vector
	double x_min = 0.0, y_min = 0.0, x_max = 0.0, y_max = 0.0; // mesh bounds
        Vec x, b;
        Mat A;
         KSP ksp;
        PC pc;
         PetscErrorCode ierr;
         PetscInt its;
};
