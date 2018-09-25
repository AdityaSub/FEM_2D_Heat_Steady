#include<iostream>
#include<iomanip>
#include<vector>
#include<fstream>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
#include<math.h>
#include<cmath>
#include "Mesh.h"

#define PI 3.14159265358979323846

using namespace std;
using namespace Eigen;

// constructor
Mesh::Mesh(string& fileName) : meshFileName(fileName) {
	readMesh();
	for (size_t i = 0; i < static_cast<size_t>(elements.size()); i++) {
		const Node n1(elements[i][0], nodes[elements[i][0]][0], nodes[elements[i][0]][1]);
		const Node n2(elements[i][1], nodes[elements[i][1]][0], nodes[elements[i][1]][1]);
		const Node n3(elements[i][2], nodes[elements[i][2]][0], nodes[elements[i][2]][1]);
		mesh.push_back(Element(n1, n2, n3, i));		
	}
	cout << "Grid generated! Bounds: x_min = " << x_min << ", y_min = " << y_min << ", x_max = " << x_max << ", y_max = " << y_max << ", nodes = " << nodes.size() << ", elements = " << elements.size() << endl;
}

// read mesh-data from file
void Mesh::readMesh() {
	ifstream mesh_read;
	mesh_read.open(meshFileName);
	// read node coordinates
	int n_nodes, n_dim; // number of nodes, number of dimensions	
	mesh_read >> n_nodes >> n_dim;
	nodes.resize(n_nodes);
	for (int i = 0; i < n_nodes; i++) {
		nodes[i].resize(n_dim);
	}
	for (int i = 0; i < n_nodes; i++) {
		for (int j = 0; j < n_dim; j++) {
			mesh_read >> nodes[i][j];
		}
	}
	
	// read connectivities
	int n_elems, n_nodes_per_elem;
	mesh_read >> n_elems >> n_nodes_per_elem; // number of elements, number of nodes per element
	elements.resize(n_elems);
	for (int i = 0; i < n_elems; i++) {
		elements[i].resize(n_nodes_per_elem);
	}
	for (int i = 0; i < n_elems; i++) {
		for (int j = 0; j < n_nodes_per_elem; j++) {
			mesh_read >> elements[i][j];
			elements[i][j] -= 1; // '0' - indexing
		}
	}
	mesh_read.close();

	for (int i = 0; i < n_nodes; i++) {
		if (x_max < nodes[i][0])
			x_max = nodes[i][0];
		if (x_min > nodes[i][0])
			x_min = nodes[i][0];
		if (y_max < nodes[i][1])
			y_max = nodes[i][1];
		if (y_min > nodes[i][1])
			y_min = nodes[i][1];
	}
}

// return reference to grid
const vector<Element>& Mesh::getGrid() const {
	return mesh;
}

// assemble global stiffness matrix
void Mesh::Assemble() {
	double num_elems = elements.size();
	// initialize global stiffness matrix
	/*globalStiffness.resize(nodes.size());
	for (int i = 0; i < nodes.size(); i++) {
		globalStiffness[i].resize(nodes.size());
		for (vector<double>::iterator it = globalStiffness[i].begin(); it != globalStiffness[i].end(); it++)
			*it = 0.0;
	}
	globalForce.resize(nodes.size());
        //x.resize(nodes.size());
        for (int i = 0; i < nodes.size(); i++) {
                globalForce[i] = 0.0;
                x(i) = 0.0;
        }*/

	cout << "global stiffness matrix, global force-vector, solution-vector initialized!" << endl;

        vector<int> nodeIDs;
	nodeIDs.resize(3);
	array<array<double, 2>, 3> elemCoords;
        double factor = 1/3.0;
        array<double, 3> gauss_pt_weights = { factor, factor, factor };
        Matrix<double, 3, 2> gauss_pts;
        gauss_pts << 0.5, 0.0, 0.0, 0.5, 0.5, 0.5;
        array<double, 3> basis_values;
        double x_query = 0.0; double y_query = 0.0;
        
        PetscInt indices1M[3];
        PetscScalar zero = 0.0, one = 1.0, force_calc, Dirichlet_value;
        PetscReal norm, tol = 1.e-14;    
    
        ierr = MatCreate(PETSC_COMM_WORLD,&A);
        ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,nodes.size(),nodes.size());
        ierr = MatSetUp(A);
        ierr = MatZeroEntries(A);       
      
        ierr = VecCreate(PETSC_COMM_WORLD,&x);
        ierr = VecSetSizes(x,PETSC_DECIDE,nodes.size());
        ierr = VecSetFromOptions(x);
        ierr = VecSet(x,zero);
  
        ierr = VecCreate(PETSC_COMM_WORLD,&b);
        ierr = VecSetSizes(b,PETSC_DECIDE,nodes.size());
        ierr = VecSetFromOptions(b);

        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);
        ierr = KSPSetOperators(ksp,A,A);
        ierr = KSPGetPC(ksp,&pc);
        ierr = PCSetType(pc,PCJACOBI);
        ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
        ierr = KSPSetFromOptions(ksp);
        
	// loop through elements and add up contributions (except at Dirichlet boundary nodes)
	for (vector<Element>::iterator it = mesh.begin(); it != mesh.end(); it++) { 
                //cout << endl;
		int elemID = it->getElemID();
		nodeIDs[0] = it->getNode1().getID(); 
		nodeIDs[1] = it->getNode2().getID();
		nodeIDs[2] = it->getNode3().getID();
		elemCoords[0] = it->getNode1().getCoord();
		elemCoords[1] = it->getNode2().getCoord();
		elemCoords[2] = it->getNode3().getCoord();
                //cout << "elemID: " << elemID << "n1 n2 n3 " << nodeIDs[0] << " " << nodeIDs[1] << " " << nodeIDs[2] << endl; 
                //cout << "elemID: " << elemID << " c1: (" << elemCoords[0][0] << ", "  << elemCoords[0][1] << ")" << " c2: (" << elemCoords[1][0] << ", " << elemCoords[1][1] << ")" << " c3: (" << elemCoords[2][0] << ", " << elemCoords[2][1] << ")" << endl;
                it->calcElementStiffness();
                //it->printElemStiffness();
		const array<array<double, 3>, 3> elemStiffness = it->getElemStiffness();
		for (int i = 0; i < 3; i++) { 
			ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);
                        ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);  
			if ((elemCoords[i][0] == x_min) || (elemCoords[i][0] == x_max) || (elemCoords[i][1] == y_min) || (elemCoords[i][1] == y_max)) { // left, right, bottom, top boundaries
                        //if ((elemCoords[i][0] == x_min) || (elemCoords[i][0] == x_max) || (elemCoords[i][1] == y_min)) { // left, right, bottom boundaries
                        //if (std::abs(sqrt(pow(elemCoords[i][0], 2.0) + pow(elemCoords[i][1], 2.0)) - 1.0) < 1e-3) {
				//globalStiffness[nodeIDs[i]][nodeIDs[i]] = 1.0;
				//globalForce[nodeIDs[i]] = 0.0;
				ierr = MatSetValues(A,1,&nodeIDs[i],1,&nodeIDs[i],&one,INSERT_VALUES); 
			/*	if ((elemCoords[i][0] > 0) && (elemCoords[i][1] >= 0)) {
					Dirichlet_value = sin(3*atan((elemCoords[i][1])/elemCoords[i][0]));
				}
				else if ((elemCoords[i][0] < 0) && (elemCoords[i][1] >= 0)) {
					Dirichlet_value = sin(3*(PI - atan((elemCoords[i][1])/std::abs(elemCoords[i][0]))));
				}
				else if ((elemCoords[i][0] < 0) && (elemCoords[i][1] <= 0)) {
					Dirichlet_value = sin(3*(PI + atan(std::abs((elemCoords[i][1])/elemCoords[i][0]))));
				}
				else if ((elemCoords[i][0] > 0) && (elemCoords[i][1] <= 0)) {
					Dirichlet_value = sin(3*(2*PI - atan(std::abs(elemCoords[i][1])/elemCoords[i][0])));
				}
				else if ((elemCoords[i][0] == 0) && (elemCoords[i][1] >= 0)) {
					Dirichlet_value = sin(3*PI/2);
				}
				else if ((elemCoords[i][0] == 0) && (elemCoords[i][1] < 0)) {
					Dirichlet_value = sin(9*PI/2);
				}*/
				ierr = VecSetValues(b,1,&nodeIDs[i],&zero,INSERT_VALUES); 				
				/*ierr = VecSetValues(b,1,&nodeIDs[i],&Dirichlet_value,INSERT_VALUES); 
			}*/
			/*else if (std::abs(sqrt(pow(elemCoords[i][0], 2.0) + pow(elemCoords[i][1], 2.0)) - 0.2) < 1e-3) {
				Dirichlet_value = 2.0;
				ierr = MatSetValues(A,1,&nodeIDs[i],1,&nodeIDs[i],&one,INSERT_VALUES);
				ierr = VecSetValues(b,1,&nodeIDs[i],&Dirichlet_value,INSERT_VALUES); 
			}*/
                        /*else if (elemCoords[i][1] == y_max) { // top boundary
                                //globalStiffness[nodeIDs[i]][nodeIDs[i]] = 1.0;
                                //globalForce[nodeIDs[i]] = 1.0;
                                ierr = MatSetValues(A,1,&nodeIDs[i],1,&nodeIDs[i],&one,INSERT_VALUES); 
			        ierr = VecSetValues(b,1,&nodeIDs[i],&one,INSERT_VALUES);
                        }*/
                        }
			else {
				ierr = MatAssemblyBegin(A,MAT_FLUSH_ASSEMBLY);
                                ierr = MatAssemblyEnd(A,MAT_FLUSH_ASSEMBLY);                
                
				for (int j = 0; j < 3; j++) {
					//globalStiffness[nodeIDs[i]][nodeIDs[j]] += elemStiffness[i][j];
					ierr = MatSetValues(A,1,&nodeIDs[i],1,&nodeIDs[j],&elemStiffness[i][j],ADD_VALUES);
                                        //cout << "Ag for " << nodeIDs[i] << ", " << nodeIDs[j] << ": " << globalStiffness[nodeIDs[i]][nodeIDs[j]] << endl;
				}				
                                //globalForce[nodeIDs[i]] = 0.0;
                                //ierr = VecSetValues(b,1,&nodeIDs[i],&zero,INSERT_VALUES);
                                for (int k = 0; k < gauss_pts.rows(); k++) {
                                        basis_values = it->getBasis().calcBasis(gauss_pts(k, 0), gauss_pts(k, 1));
                                        //globalForce[nodeIDs[i]] += (-0.5) * gauss_pt_weights[k] * basis_values[i] * it->getBasis().getDetJ();                                        
                                        x_query = 0.0; y_query = 0.0;
                                        for (int l = 0; l < basis_values.size(); l++) {
                                                x_query += basis_values[l] * elemCoords[l][0];
												y_query += basis_values[l] * elemCoords[l][1];
                                        }
                                        //cout << "elemID: " << elemID << " gp" << k << ": (" << x_query << ", " << y_query << ")" << endl;
                                        //globalForce[nodeIDs[i]] += 0.5 * gauss_pt_weights[k] * basis_values[i] * (-2) * pow(PI, 2.0) * sin(PI * x_query) * sin(PI * y_query) * it->getBasis().getDetJ();
                                        force_calc = 0.5 * gauss_pt_weights[k] * basis_values[i] * (-2) * pow(PI, 2.0) * sin(PI * x_query) * sin(PI * y_query) * it->getBasis().getDetJ();
				        ierr = VecSetValues(b,1,&nodeIDs[i],&force_calc,ADD_VALUES);
                                }
			}
		}
	}
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    //ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);
   
    ierr = VecAssemblyBegin(b);
    ierr = VecAssemblyEnd(b);
    //ierr = VecView(b,PETSC_VIEWER_STDOUT_SELF); 
}

void Mesh::Solve() {
	/*MatrixXd matGlobalSti(globalStiffness.size(), globalStiffness[0].size());
       	VectorXd vecGlobalFor = Map<VectorXd, Unaligned>(globalForce.data(), globalForce.size());
        for (int i = 0; i < nodes.size(); i++) {
		matGlobalSti.row(i) = VectorXd::Map(&globalStiffness[i][0], globalStiffness[i].size());
	}
       	//cout << matGlobalSti << endl;
        //cout << vecGlobalFor << endl;
	cout << "Solving...";
	x = matGlobalSti.householderQr().solve(vecGlobalFor);
        //x = matGlobalSti.lu().solve(vecGlobalFor);
	double relative_error = (matGlobalSti*x - vecGlobalFor).norm() / vecGlobalFor.norm(); // relative error
	cout << "done! Relative error = " << relative_error << endl;*/
	//cout << x << endl;
	ierr = KSPSetFromOptions(ksp);
    ierr = KSPSolve(ksp,b,x);
    ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);
    ierr = KSPGetIterationNumber(ksp,&its);	   
}

void Mesh::writeField(const string& solName) {
      ofstream outFile;           
      stringstream ss;
      ss << solName << ".plt";
      outFile.open(ss.str());
      outFile << "TITLE = \"SOLUTION_FIELD\"" << endl;
      outFile << "VARIABLES = \"x\" \"y\" \"T\"" << endl;
      outFile << "ZONE N=" << nodes.size() << ", E=" << elements.size() << ", F=FEPOINT, ET=TRIANGLE" << endl;
      x_sol.resize(nodes.size());
      for (int i=0; i<nodes.size(); i++) {
	      VecGetValues(x,1,&i,&x_sol(i));
              outFile << setprecision(6) << nodes[i][0] << "\t" << setprecision(6) << nodes[i][1] << "\t" << setprecision(6) << x_sol(i) << endl;
      }

      for (int i=0; i<elements.size(); i++) {
              outFile << elements[i][0] + 1 << "\t" << elements[i][1] + 1 << "\t" << elements[i][2] + 1 << endl;
      }
      outFile.close();        
}

void Mesh::computeL2Error() {
      vector<int> nodeIDs;
      nodeIDs.resize(3);
      array<array<double, 2>, 3> elemCoords;
      double factor = 1/3.0;
      array<double, 3> gauss_pt_weights = { factor, factor, factor };
      Matrix<double, 3, 2> gauss_pts;
      gauss_pts << 0.5, 0.0, 0.0, 0.5, 0.5, 0.5;
      array<double, 3> basis_values;
      array<double, 3> elem_dofs;
      double x_query = 0.0, y_query = 0.0, dof_val = 0.0;
      double L2_error = 0.0;
      for (vector<Element>::iterator it = mesh.begin(); it != mesh.end(); it++) {
              nodeIDs[0] = it->getNode1().getID();
              nodeIDs[1] = it->getNode2().getID();
              nodeIDs[2] = it->getNode3().getID();
              elemCoords[0] = it->getNode1().getCoord();
              elemCoords[1] = it->getNode2().getCoord();
              elemCoords[2] = it->getNode3().getCoord();
              elem_dofs[0] = x_sol(nodeIDs[0]);
              elem_dofs[1] = x_sol(nodeIDs[1]);
              elem_dofs[2] = x_sol(nodeIDs[2]);
              for (int k = 0; k < gauss_pts.rows(); k++) {
                      basis_values = it->getBasis().calcBasis(gauss_pts(k, 0), gauss_pts(k, 1));
                      x_query = 0.0; y_query = 0.0; dof_val = 0.0;
                      for (int l = 0; l < basis_values.size(); l++) {
                              x_query += basis_values[l] * elemCoords[l][0];
                              y_query += basis_values[l] * elemCoords[l][1];
                              dof_val += basis_values[l] * elem_dofs[l];
                      }
                      L2_error += 0.5 * gauss_pt_weights[k] * pow(dof_val - sin(PI * x_query) * sin(PI * y_query), 2.0) * it->getBasis().getDetJ();
              }
      }
      L2_error = sqrt(L2_error);
      cout << "L2 error: " << L2_error << endl;
}

// destructor
Mesh::~Mesh() { 	 
    ierr = MatDestroy(&A);
    ierr = VecDestroy(&x);  
    ierr = VecDestroy(&b);
    ierr = KSPDestroy(&ksp);
    ierr = PetscFinalize();
    cout << "Grid destroyed!" << endl; 
}
