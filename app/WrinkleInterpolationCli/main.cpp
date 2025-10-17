#include "polyscope/polyscope.h"
#include "polyscope/pick.h"

#include <igl/invert_diag.h>
#include <igl/per_vertex_normals.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/loop.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/cotmatrix_entries.h>

#include <iostream>
#include <filesystem>
#include <utility>
#include <CLI/CLI.hpp>

#include "../../include/CommonTools.h"
#include "../../include/MeshLib/MeshUpsampling.h"
#include "../../include/Optimization/NewtonDescent.h"
#include "../../include/IntrinsicFormula/InterpolateZvals.h"
#include "../../include/IntrinsicFormula/WrinkleEditingModel.h"
#include "../../include/IntrinsicFormula/WrinkleEditingCWF.h"

#include "../../include/IntrinsicFormula/KnoppelStripePatternEdgeOmega.h"
#include "../../include/WrinkleFieldsEditor.h"
#include "../../dep/SecStencils/types.h"
#include "../../dep/SecStencils/Subd.h"
#include "../../dep/SecStencils/utils.h"

#include "../../include/json.hpp"
#include "../../include/ComplexLoop/ComplexLoop.h"
#include "../../include/ComplexLoop/ComplexLoopZuenko.h"
#include "../../include/LoadSaveIO.h"
#include "../../include/SecMeshParsing.h"
#include "../../include/MeshLib/RegionEdition.h"

std::vector<VertexOpInfo> vertOpts;

std::vector<Eigen::MatrixXd> triV, upsampledTriV;
Eigen::MatrixXi triF, upsampledTriF;
MeshConnectivity triMesh;
Mesh secMesh, subSecMesh;

// initial information
Eigen::VectorXd initAmp;
Eigen::VectorXd initOmega;
std::vector<std::complex<double>> initZvals;

// target information
Eigen::VectorXd tarAmp;
Eigen::VectorXd tarOmega;
std::vector<std::complex<double>> tarZvals;

// base mesh information list
std::vector<Eigen::VectorXd> ampList;
std::vector<Eigen::VectorXd> omegaList;

std::vector<Eigen::MatrixXd> faceOmegaList2Dinput;
std::vector<Eigen::MatrixXd> faceOmegaList3Dinput;

std::vector<Eigen::MatrixXd> vertexOmegaList;

std::vector<Eigen::MatrixXd> faceOmegaList;
std::vector<std::vector<std::complex<double>>> zList;


// upsampled informations
std::vector<Eigen::VectorXd> subOmegaList;
std::vector<Eigen::MatrixXd> subFaceOmegaList;

std::vector<Eigen::VectorXd> phaseFieldsList;
std::vector<Eigen::VectorXd> ampFieldsList;
std::vector<std::vector<std::complex<double>>> upZList;
std::vector<Eigen::MatrixXd> wrinkledVList;


// region edition
RegionEdition regEdt;

int numFrames = 51;

double globalAmpMax = 1;
double globalAmpMin = 0;

double globalInconMax = 1;
double globalInconMin = 0;

double globalCoarseInconMax = 1;
double globalCoarseInconMin = 0;

int quadOrder = 4;

double spatialAmpRatio = 1000;
double spatialEdgeRatio = 1000;
double spatialKnoppelRatio = 1000;

std::string workingFolder;

std::shared_ptr<IntrinsicFormula::WrinkleEditingModel> editModel;

bool isFixedBnd = false;
int effectivedistFactor = 4;

bool isSelectAll = false;
VecMotionType selectedMotion = Enlarge;

double selectedMotionValue = 2;
double selectedMagValue = 1;
bool isCoupled = false;

Eigen::VectorXi selectedFids;
Eigen::VectorXi interfaceFids;
Eigen::VectorXi faceFlags;	// -1 for interfaces, 0 otherwise
Eigen::VectorXi selectedVertices;

int optTimes = 5;

bool isLoadOpt = false;
bool isLoadTar = false;

int clickedFid = -1;
int dilationTimes = 10;

bool isUseV2 = false;
int upsampleTimes = 3;


// Default arguments
struct {
	std::string input;
	std::string method = "CWF";
	double gradTol = 1e-6;
	double xTol = 0;
	double fTol = 0;
	int numIter = 1000;
	double ampScale = 1;
	bool reOptimize = false;
} args;

using Complex = std::complex<double>;

// --- Utility: Get Z-values from Eigen Matrix ---
// Note: Assuming Z values are stored in a structure compatible with the application's list format
// (std::vector<std::complex<double>>)

void copyComplexEigenToStdVector(
    const Eigen::VectorXcd& psi_eigen, 
    std::vector<Complex>& zvals_std)
{
    int nverts = psi_eigen.size();
    zvals_std.resize(nverts);
    for (int i = 0; i < nverts; ++i) {
        zvals_std[i] = psi_eigen(i);
    }
}

// --- Utility: Get Z-values from Std Vector to Eigen Matrix ---

void copyComplexStdVectorToEigen(
    const std::vector<Complex>& zvals_std,
    Eigen::VectorXcd& psi_eigen)
{
    int nverts = zvals_std.size();
    psi_eigen.resize(nverts);
    for (int i = 0; i < nverts; ++i) {
        psi_eigen(i) = zvals_std[i];
    }
}


InitializationType initType = SeperateLinear;
double zuenkoTau = 0.1;
int zuenkoIter = 5;

static void buildEditModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor, std::shared_ptr<IntrinsicFormula::WrinkleEditingModel>& editModel)
{
	editModel = std::make_shared<IntrinsicFormula::WrinkleEditingCWF>(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor);
}

Eigen::Vector3d lineFieldAddition(const std::vector<Eigen::Vector3d>& grad_phi_list) {
    if (grad_phi_list.empty()) {
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d total_vec = Eigen::Vector3d::Zero();

    for (const auto& vec : grad_phi_list) {
        // Assume the vector is approximately in the X-Z plane (common in wrinkle fields)
        double norm = vec.norm();
        // atan2(y, x) -> using Z as the vertical component (y) and X as the horizontal (x)
        double angle = std::atan2(vec[2], vec[0]);

        // Map to double-angle space: [norm*cos(2*angle), 0, norm*sin(2*angle)]
        total_vec[0] += norm * std::cos(2.0 * angle); // X component of doubled vector
        // total_vec[1] += 0.0;                       // Y component remains 0
        total_vec[2] += norm * std::sin(2.0 * angle); // Z component of doubled vector
    }

    // Average the doubled vector
    total_vec /= (double)grad_phi_list.size();

    // Map back to single-angle space
    double final_norm = total_vec.norm();
    // atan2(y, x) -> using Z as y, X as x
    double final_angle = std::atan2(total_vec[2], total_vec[0]);

    // Halve the angle: final_angle / 2
    Eigen::Vector3d final_vec;
    final_vec[0] = final_norm * std::cos(final_angle / 2.0); // X component
    final_vec[1] = 0.0;                                     // Y component
    final_vec[2] = final_norm * std::sin(final_angle / 2.0); // Z component

    return final_vec;
}

void getGradPhisPerVertexLineField(const Eigen::MatrixXd& face_grad_phis, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& vertex_grad_phis) 
{
    int num_vertices = V.rows();
    int num_faces = F.rows();
    
    // vertex_phis_list: Stores a list of face omega vectors adjacent to each vertex
    std::vector<std::vector<Eigen::Vector3d>> vertex_phis_list(num_vertices);

    // 1. Accumulate face vectors per vertex
    for (int face_idx = 0; face_idx < num_faces; ++face_idx) {
        Eigen::Vector3d face_vec = face_grad_phis.row(face_idx).transpose();
        
        for (int k = 0; k < 3; ++k) {
            int vertex_idx = F(face_idx, k);
            vertex_phis_list[vertex_idx].push_back(face_vec);
        }
    }

    // 2. Perform Line Field Averaging
    vertex_grad_phis.resize(num_vertices, 3);
    
    for (int i = 0; i < num_vertices; ++i) {
        // Use the lineFieldAddition helper function
        Eigen::Vector3d averaged_vec = lineFieldAddition(vertex_phis_list[i]);
        vertex_grad_phis.row(i) = averaged_vec.transpose();
    }
}

void initializeRandomUnitNormZvals(const Eigen::MatrixXd& triV) 
{
    // 1. Determine the number of vertices
    int nverts = triV.rows();
    
    // 2. Resize the container
    initZvals.resize(nverts);

    // 3. Setup Random Number Generator
    // Use a high-quality pseudo-random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Define a distribution for the angle (phase) from 0 to 2*pi
    std::uniform_real_distribution<> distrib(0.0, 2.0 * M_PI); // M_PI is typically defined in cmath or <numbers>

    // 4. Populate initZvals
    for (int i = 0; i < nverts; ++i) {
        // Generate a random angle (phi)
        double angle = distrib(gen);
        
        // Magnitude (norm) is fixed at 1.0 for unit norm
        double magnitude = 1.0; 
        
        // Convert polar coordinates (magnitude, angle) to complex number (a + bi)
        // Re(Z) = magnitude * cos(angle)
        // Im(Z) = magnitude * sin(angle)
        double real_part = magnitude * std::cos(angle);
        double imag_part = magnitude * std::sin(angle);
        
        initZvals[i] = std::complex<double>(real_part, imag_part);
    }
    
    std::cout << "Initialized " << nverts << " Z-values with random unit norm." << std::endl;
}

void updateMagnitudePhase(const std::vector<Eigen::VectorXd>& wFrames, const std::vector<std::vector<std::complex<double>>>& zFrames, 
	std::vector<Eigen::VectorXd>& magList, 
	std::vector<Eigen::VectorXd>& phaseList,
	std::vector<std::vector<std::complex<double>>>& upZFrames)
{
	magList.resize(wFrames.size());
	phaseList.resize(wFrames.size());
	upZFrames.resize(wFrames.size());

	subOmegaList.resize(wFrames.size());
	subFaceOmegaList.resize(wFrames.size());
	
	MeshConnectivity mesh(triF);


	auto computeMagPhase = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			Eigen::VectorXd edgeVec = swapEdgeVec(triF, wFrames[i], 0);

			std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
			complexLoopOpt->setBndFixFlag(isFixedBnd);
			complexLoopOpt->SetMesh(secMesh);
			complexLoopOpt->Subdivide(edgeVec, zFrames[i], subOmegaList[i], upZFrames[i], upsampleTimes);
			Mesh tmpMesh = complexLoopOpt->GetMesh();

			subFaceOmegaList[i] = edgeVec2FaceVec(tmpMesh, subOmegaList[i]);
			
			magList[i].setZero(upZFrames[i].size());
			phaseList[i].setZero(upZFrames[i].size());

			for (int j = 0; j < magList[i].size(); j++)
			{
				magList[i](j) = std::abs(upZFrames[i][j]);
				phaseList[i](j) = std::arg(upZFrames[i][j]);
			}
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)upZFrames.size());
	tbb::parallel_for(rangex, computeMagPhase);

}


void updateWrinkles(const std::vector<Eigen::MatrixXd>& V, const Eigen::MatrixXi& F, const std::vector<std::vector<std::complex<double>>>& zFrames, std::vector<Eigen::MatrixXd>& wrinkledVFrames, double scaleRatio, bool isUseV2)
{
	std::vector<std::vector<std::complex<double>>> interpZList(zFrames.size());
	wrinkledVFrames.resize(zFrames.size());

	std::vector<std::vector<int>> vertNeiEdges;
	std::vector<std::vector<int>> vertNeiFaces;

	buildVertexNeighboringInfo(MeshConnectivity(F), V[0].rows(), vertNeiEdges, vertNeiFaces);

	auto computeWrinkles = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			getWrinkledMesh(V[i], F, zFrames[i], &vertNeiFaces, wrinkledVFrames[i], scaleRatio, isUseV2);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)interpZList.size());
	tbb::parallel_for(rangex, computeWrinkles);

}

void updateEverythingForSaving()
{
	// get interploated amp and phase frames
	// this is fine, because it does the processing on the base mesh like i need.
	std::cout << "compute upsampled phase: " << std::endl;
	updateMagnitudePhase(omegaList, zList, ampFieldsList, phaseFieldsList, upZList);

	std::cout << "compute wrinkle meshes: " << std::endl;
	//wrinkledVList contains all the frames for the mesh with displaced wrinkles
	updateWrinkles(upsampledTriV, upsampledTriF, upZList, wrinkledVList, args.ampScale, isUseV2);

	//these are actually not used anywhere, i suppose only for visualisation? don't really need
	//hence, doing it over the base mesh for now
	std::cout << "compute face vector fields:" << std::endl;
	faceOmegaList.resize(omegaList.size());
	for (int i = 0; i < omegaList.size(); i++)
	{
		faceOmegaList[i] = intrinsicEdgeVec2FaceVec(omegaList[i], triV[0], triMesh);
	}
}

void getUpsampledMesh(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	secMesh = convert2SecMesh(triV, triF);
	subSecMesh = secMesh;

	std::shared_ptr<ComplexLoop> complexLoopOpt = std::make_shared<ComplexLoopZuenko>();
	complexLoopOpt->setBndFixFlag(isFixedBnd);
	complexLoopOpt->SetMesh(secMesh);
	complexLoopOpt->meshSubdivide(upsampleTimes);
	subSecMesh = complexLoopOpt->GetMesh();
	parseSecMesh(subSecMesh, upsampledTriV, upsampledTriF);
}

void initialization(const Eigen::MatrixXd& triV, const Eigen::MatrixXi& triF, Eigen::MatrixXd& upsampledTriV, Eigen::MatrixXi& upsampledTriF)
{
	getUpsampledMesh(triV, triF, upsampledTriV, upsampledTriF);
	selectedFids.setZero(triMesh.nFaces());
	interfaceFids = selectedFids;
	regEdt = RegionEdition(triMesh, triV.rows());
	selectedVertices.setZero(triV.rows());
}

// bool loadProblem()
// {
// 	std::string loadFileName = args.input;
// 	std::cout << "load file in: " << loadFileName << std::endl;
// 	using json = nlohmann::json;
// 	std::ifstream inputJson(loadFileName);
// 	if (!inputJson) {
// 		std::cerr << "missing json file in " << loadFileName << std::endl;
// 		return false;
// 	}

// 	std::string filePath = loadFileName;
// 	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
// 	int id = filePath.rfind("/");
// 	workingFolder = filePath.substr(0, id + 1);
// 	std::cout << "working folder: " << workingFolder << std::endl;

// 	json jval;
// 	inputJson >> jval;

// 	std::string meshFile = jval["mesh_name"];
// 	upsampleTimes = jval["upsampled_times"];


// 	meshFile = workingFolder + meshFile;
// 	igl::readOBJ(meshFile, triV, triF);
// 	triMesh = MeshConnectivity(triF);
// 	initialization(triV, triF, upsampledTriV, upsampledTriF);
	

// 	quadOrder = jval["quad_order"];
// 	numFrames = jval["num_frame"];
//     if (jval.contains(std::string_view{ "wrinkle_amp_ratio" }))
//     {
//         if(args.ampScale == 1)
//             args.ampScale = jval["wrinkle_amp_ratio"];
//     }

// 	isSelectAll = jval["region_global_details"]["select_all"];
// 	isCoupled = jval["region_global_details"]["amp_omega_coupling"];
// 	selectedMagValue = jval["region_global_details"]["amp_operation_value"];
// 	selectedMotionValue = jval["region_global_details"]["omega_operation_value"];
// 	std::string optype = jval["region_global_details"]["omega_operation_motion"];

// 	if (optype == "None")
// 		selectedMotion = None;
// 	else if (optype == "Enlarge")
// 		selectedMotion = Enlarge;
// 	else if (optype == "Rotate")
// 		selectedMotion = Rotate;
// 	else
// 		selectedMotion = None;

// 	pickFaces.clear();


// 	int nedges = triMesh.nEdges();
// 	int nverts = triV.rows();

// 	std::string initAmpPath = jval["init_amp"];
// 	std::string initOmegaPath = jval["init_omega"];
// 	std::string initZValsPath = "zvals.txt";
// 	if (jval.contains(std::string_view{ "init_zvals" }))
// 	{
// 		initZValsPath = jval["init_zvals"];
// 	}

// 	if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, initOmega)) {
// 		std::cout << "missing init edge omega file." << std::endl;
// 		return false;
// 	}

// 	if (!loadVertexZvals(workingFolder + initZValsPath, triV.rows(), initZvals))
// 	{
// 		std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
// 		if (!loadVertexAmp(workingFolder + initAmpPath, triV.rows(), initAmp))
// 		{
// 			std::cout << "missing init amp file: " << std::endl;
// 			return false;
// 		}

// 		else
// 		{
// 			Eigen::VectorXd edgeArea, vertArea;
// 			edgeArea = getEdgeArea(triV, triMesh);
// 			vertArea = getVertArea(triV, triMesh);
// 			IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, initOmega, initAmp, edgeArea, vertArea, triV.rows(), initZvals);
// 		}
// 	}
// 	else
// 	{
// 		initAmp.setZero(triV.rows());
// 		for (int i = 0; i < initZvals.size(); i++)
// 			initAmp(i) = std::abs(initZvals[i]);
// 	}

// 	std::string optZvals = jval["solution"]["opt_zvals"];
// 	std::string optOmega = jval["solution"]["opt_omega"];



// 	isLoadOpt = true;
// 	zList.clear();
// 	omegaList.clear();
// 	ampList.clear();
// 	for (int i = 0; i < numFrames; i++)
// 	{
// 		std::string zvalFile = workingFolder + optZvals + "/zvals_" + std::to_string(i) + ".txt";
// 		std::string edgeOmegaFile = workingFolder + optOmega + "/omega_" + std::to_string(i) + ".txt";
		
// 		std::vector<std::complex<double>> zvals;
// 		Eigen::VectorXd vertAmp;

// 		if (!loadVertexZvals(zvalFile, nverts, zvals))
// 		{
// 			isLoadOpt = false;
// 			break;
// 		}
// 		else {
// 			vertAmp.setZero(zvals.size());
// 			for (int i = 0; i < zvals.size(); i++) {
// 				vertAmp[i] = std::abs(zvals[i]);
// 			}
// 		}

// 		Eigen::VectorXd edgeOmega;
// 		if (!loadEdgeOmega(edgeOmegaFile, nedges, edgeOmega)) {
// 			isLoadOpt = false;
// 			break;
// 		}

// 		zList.push_back(zvals);
// 		omegaList.push_back(edgeOmega);
// 		ampList.push_back(vertAmp);
// 	}

// 	if (isLoadOpt)
// 	{
// 		std::cout << "load zvals and omegas from file!" << std::endl;
// 	}
// 	if (!isLoadOpt)
// 	{
// 		buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);

//         if(!isLoadTar)
//         {
//             editModel->editCWFBasedOnVertOp(initZvals, initOmega, tarZvals, tarOmega);
//         }
// 		editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, numFrames - 2, true);

// 		zList = editModel->getVertValsList();
// 		omegaList = editModel->getWList();
// 		ampList = editModel->getRefAmpList();

// 	}
// 	else
// 	{
// 		buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);
// 		editModel->initialization(zList, omegaList, ampList, omegaList);
// 	}
	

// 	return true;
// }

bool readFaces(std::string &facesFile) 
{
    std::ifstream file(facesFile);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open faces file: " << facesFile << std::endl;
        return false;
    }

    std::vector<int> face_indices;
    int num_faces = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string cell;
        int count = 0;
        
        while (std::getline(ss, cell, ',')) {
            try {
                face_indices.push_back(std::stoi(cell));
                count++;
            } catch (const std::exception& e) {
                std::cerr << "Error reading face index: " << e.what() << std::endl;
                return false;
            }
        }
        
        if (count != 3) {
            std::cerr << "Error: Face line does not contain 3 indices." << std::endl;
            // Depending on strictness, you might return false or skip.
            // Returning false for strict compliance.
            return false;
        }
        num_faces++;
    }

    if (num_faces == 0) {
        triF.resize(0, 3);
        return true;
    }

    // Resize and map the flattened vector to the Eigen matrix
    triF.resize(num_faces, 3);
    for (int i = 0; i < num_faces; ++i) {
        triF(i, 0) = face_indices[i * 3];
        triF(i, 1) = face_indices[i * 3 + 1];
        triF(i, 2) = face_indices[i * 3 + 2];
    }
    
    std::cout << "Successfully read " << num_faces << " faces." << std::endl;
    return true;
}

bool readVertices(std::string &verticesFile, Eigen::MatrixXd &triV)
{
    std::ifstream file(verticesFile);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open vertices file: " << verticesFile << std::endl;
        return false;
    }

    std::vector<std::vector<double>> coords;
    coords.resize(3);
    
    int num_lines = 0;
    std::string line;
    int vertex_count = -1;

    while (std::getline(file, line) && num_lines < 3) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string cell;
        int current_count = 0;
        
        while (std::getline(ss, cell, ',')) {
            try {
                // Remove potential carriage return (\r) if reading files created on Windows
                if (!cell.empty() && cell.back() == '\r') {
                    cell.pop_back();
                }
                coords[num_lines].push_back(std::stod(cell));
                current_count++;
            } catch (const std::exception& e) {
                std::cerr << "Error reading coordinate: " << e.what() << std::endl;
                return false;
            }
        }
        
        if (vertex_count == -1) {
            vertex_count = current_count;
        } else if (vertex_count != current_count) {
            std::cerr << "Error: Coordinate lines have inconsistent vertex counts." << std::endl;
            return false;
        }
        
        num_lines++;
    }

    if (num_lines != 3) {
        std::cerr << "Error: Vertices file must contain exactly 3 coordinate lines (X, Y, Z)." << std::endl;
        return false;
    }

    if (vertex_count == 0) {
        triV.resize(0, 3);
        return true;
    }

    triV.resize(vertex_count, 3); 
    for (int j = 0; j < vertex_count; ++j) {
        triV(j, 0) = coords[0][j]; // X
        triV(j, 1) = coords[1][j]; // Y
        triV(j, 2) = coords[2][j]; // Z
    }
    
    std::cout << "Successfully read " << vertex_count << " vertices." << std::endl;
    return true;
}

bool loadFaceOmegas(const std::string& filePath, const int& nfaces, Eigen::MatrixXd& faceOmegaMat)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open face omegas file: " << filePath << std::endl;
        return false;
    }

    faceOmegaMat.resize(nfaces, 2);
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < nfaces) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        
        // Read the two components per face
        for (int col = 0; col < 2; ++col) {
            if (!std::getline(ss, cell, ',')) {
                std::cerr << "Error: Missing component in face omega file at line " << row + 1 << std::endl;
                return false;
            }
            try {
                faceOmegaMat(row, col) = std::stod(cell);
            } catch (const std::exception& e) {
                std::cerr << "Error converting face omega value at line " << row + 1 << ": " << e.what() << std::endl;
                return false;
            }
        }
        row++;
    }

    if (row != nfaces) {
        std::cerr << "Error: Expected " << nfaces << " lines but read " << row << std::endl;
        return false;
    }

    std::cout << "Successfully read " << nfaces << " face omegas." << std::endl;
    return true;
}

bool reconstructFaceOmegas(const Eigen::MatrixXd& faceOmegaList2Dinput, const Eigen::MatrixXd& V, Eigen::MatrixXd& faceOmegaList3Dinput, double scale = 1.0)
{
    int num_faces = triF.rows();
    if (faceOmegaList2Dinput.rows() != num_faces) {
        std::cerr << "Error: 2D omega input rows must match face count." << std::endl;
        return false;
    }

    // Resize the output matrix to store the 3D vectors (Nf x 3)
    faceOmegaList3Dinput.resize(num_faces, 3);

    for (int f_idx = 0; f_idx < num_faces; ++f_idx)
    {
        // 1. Get vertex coordinates for face (i, j, k)
        int i = triF(f_idx, 0);
        int j = triF(f_idx, 1);
        int k = triF(f_idx, 2);

        Eigen::Vector3d v0 = V.row(i).transpose();
        Eigen::Vector3d v1 = V.row(j).transpose();
        Eigen::Vector3d v2 = V.row(k).transpose();

        // 2. Define the basis vectors (edges v0->v1 and v0->v2)
        // These are the vectors onto which the 3D omega vector (psi) was projected.
        Eigen::Vector3d e1 = v1 - v0; // Edge v0->v1
        Eigen::Vector3d e2 = v2 - v0; // Edge v0->v2

        // 3. Define the system: U * psi = c
        // U is the 2x3 matrix of basis vectors (e1, e2) as rows.
        Eigen::MatrixXd U(2, 3);
        U.row(0) = e1;
        U.row(1) = e2;

        // c is the 2x1 vector of input components [c1; c2]
        Eigen::Vector2d c;
        c(0) = faceOmegaList2Dinput(f_idx, 0); // component along e1
        c(1) = faceOmegaList2Dinput(f_idx, 1); // component along e2

        // 4. Solve the least squares problem for psi (the 3D vector)
        // Since U is 2x3, we use the pseudo-inverse (Normal Equations method: U^T * U * psi = U^T * c)
        
        // A = U^T * U (3x3 matrix)
        Eigen::Matrix3d A = U.transpose() * U; 
        
        // b = U^T * c (3x1 vector)
        Eigen::Vector3d b = U.transpose() * c;

        // Solve A * psi_vec = b using LU decomposition (fast and robust for 3x3)
        // The resulting psi_vec is the minimal norm solution.
        Eigen::Vector3d psi_vec = A.lu().solve(b);

        // 5. Apply optional scaling and store
        faceOmegaList3Dinput.row(f_idx) = (psi_vec * scale).transpose();
    }
    
    std::cout << "Successfully reconstructed 3D face omegas for " << num_faces << " faces." << std::endl;
    return true;
}

void getEdgeOmegas(const Eigen::MatrixXd& V, const Eigen::MatrixXd& vertex_grad_phis, Eigen::VectorXd& edgeOmegaList)
{
    int num_edges = triMesh.nEdges();
    edgeOmegaList.resize(num_edges);

    for (int edge_id = 0; edge_id < num_edges; ++edge_id)
    {
        // Get the two vertices (i, j) connected by the edge
        int i = triMesh.edgeVertex(edge_id, 0);
        int j = triMesh.edgeVertex(edge_id, 1);
        
        // 1. Edge vector
        Eigen::Vector3d V_i = V.row(i).transpose();
        Eigen::Vector3d V_j = V.row(j).transpose();
        Eigen::Vector3d edge_vec = V_j - V_i; // Direction i -> j

        // 2. Average per-vertex omega field
        Eigen::Vector3d grad_phi_i = vertex_grad_phis.row(i).transpose();
        Eigen::Vector3d grad_phi_j = vertex_grad_phis.row(j).transpose();
        Eigen::Vector3d avg_grad_phi = 0.5 * (grad_phi_i + grad_phi_j);

        // 3. Projection (Dot Product)
        // grad_along_edge = omega_ij
        double grad_along_edge = avg_grad_phi.dot(edge_vec);
        
        // Store the result
        edgeOmegaList(edge_id) = grad_along_edge;
    }
}

bool loadSolvedProblem() {

	std::string verticesFile = "vertices.csv";
	std::string facesFile = "faces.csv";
	std::string amplitudesFile = "amplitudes.csv";
	std::string faceOmegasFile = "dphisPerFace.csv";

	facesFile = workingFolder + facesFile;
	readFaces(facesFile);
	triMesh = MeshConnectivity(triF);

	//will need to resize triV and ampList

	for(int i=0; i<numFrames; i++) {
		std::string curVerticesFile = workingFolder + verticesFile + '.' + std::to_string(i);
		readVertices(curVerticesFile, triV[i]);
		
		initialization(triV[i], triF, upsampledTriV[i], upsampledTriF);

		std::string curAmplitudesFile = workingFolder + amplitudesFile + '.' + std::to_string(i);
		loadVertexAmp(curAmplitudesFile, triV[0].rows(), ampList[i]);

		//read face omega list
		std::string curFaceOmegasFile = workingFolder + faceOmegasFile + '.' + std::to_string(i);
		loadFaceOmegas(curFaceOmegasFile, triV[0].rows(), faceOmegaList2Dinput[i]);

		reconstructFaceOmegas(faceOmegaList2Dinput[i], triV[0], faceOmegaList3Dinput[i]);

		getGradPhisPerVertexLineField(faceOmegaList3Dinput[i], triV[0], triF, vertexOmegaList[i]);

		//convert it to omegalist
		getEdgeOmegas(triV[0], vertexOmegaList[i], omegaList[i]);
	}


	return true;
}

bool saveProblem()
{
	std::string curOpt = "None";
	if (selectedMotion == Enlarge)
		curOpt = "Enlarge";
	else if (selectedMotion == Rotate)
		curOpt = "Rotate";

	using json = nlohmann::json;
	json jval =
	{
			{"mesh_name",         "mesh.obj"},
			{"num_frame",         zList.size()},
            {"wrinkle_amp_ratio", args.ampScale},
			{"quad_order",        quadOrder},
			{"spatial_ratio",     {
										   {"amp_ratio", spatialAmpRatio},
										   {"edge_ratio", spatialEdgeRatio},
										   {"knoppel_ratio", spatialKnoppelRatio}

								  }
			},
			{"upsampled_times",  upsampleTimes},
			{"init_omega",        "omega.txt"},
			{"init_amp",          "amp.txt"},
			{"init_zvals",        "zvals.txt"},
			{"tar_omega",         "omega_tar.txt"},
			{"tar_amp",           "amp_tar.txt"},
			{"tar_zvals",         "zvals_tar.txt"},
			{"region_global_details",	  {
										{"select_all", isSelectAll},
										{"omega_operation_motion", curOpt},
										{"omega_operation_value", selectedMotionValue},
										{"amp_omega_coupling", isCoupled},
										{"amp_operation_value", selectedMagValue}
								  }
			},
			{
			 "solution",          {
										  {"opt_amp", "/optAmp/"},
										  {"opt_zvals", "/optZvals/"},
										  {"opt_omega", "/optOmega/"},
										  {"wrinkle_mesh", "/wrinkledMesh/"},
										  {"upsampled_amp", "/upsampledAmp/"},
										  {"upsampled_phase", "/upsampledPhase/"}
								  }
			}
	};	

	saveEdgeOmega(workingFolder + "omega.txt", initOmega);
	saveVertexAmp(workingFolder + "amp.txt", initAmp);
	saveVertexZvals(workingFolder + "zvals.txt", initZvals);

	if (isLoadTar)
	{
		std::cout << "save target" << std::endl;
		saveEdgeOmega(workingFolder + "omega_tar.txt", tarOmega);
		saveVertexAmp(workingFolder + "amp_tar.txt", tarAmp);
		saveVertexZvals(workingFolder + "zvals_tar.txt", tarZvals);
	}

	igl::writeOBJ(workingFolder + "mesh.obj", triV[0], triF);

	std::string outputFolder = workingFolder + "/optZvals/";
	mkdir(outputFolder);

	std::string omegaOutputFolder = workingFolder + "/optOmega/";
	mkdir(omegaOutputFolder);

	std::string ampOutputFolder = workingFolder + "/optAmp/";
	mkdir(ampOutputFolder);


	int nframes = zList.size();
	auto savePerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{

			saveVertexZvals(outputFolder + "zvals_" + std::to_string(i) + ".txt", zList[i]);
			saveEdgeOmega(omegaOutputFolder + "omega_" + std::to_string(i) + ".txt", omegaList[i]);
			saveVertexAmp(ampOutputFolder + "amp_" + std::to_string(i) + ".txt", ampList[i]);
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes, GRAIN_SIZE);
	tbb::parallel_for(rangex, savePerFrame);


	std::ofstream o(args.input);
	o << std::setw(4) << jval << std::endl;
	std::cout << "save file in: " << args.input << std::endl;

	return true;
}

bool saveForRender()
{
	// render information
	std::string renderFolder = workingFolder + "/render/";
	mkdir(renderFolder);
	igl::writeOBJ(renderFolder + "basemesh.obj", triV[0], triF);
	igl::writeOBJ(renderFolder + "upmesh.obj", upsampledTriV[0], upsampledTriF);


	saveFlag4Render(faceFlags, renderFolder + "faceFlags.cvs");

	std::string outputFolderAmp = renderFolder + "/upsampledAmp/";
	mkdir(outputFolderAmp);

	std::string outputFolderPhase = renderFolder + "/upsampledPhase/";
	mkdir(outputFolderPhase);

	std::string outputFolderWrinkles = renderFolder + "/wrinkledMesh/";
	mkdir(outputFolderWrinkles);

	std::string optAmpFolder = renderFolder + "/optAmp/";
	mkdir(optAmpFolder);
	std::string optOmegaFolder = renderFolder + "/optOmega/";
	mkdir(optOmegaFolder);

	int nframes = ampFieldsList.size();

	auto savePerFrame = [&](const tbb::blocked_range<uint32_t>& range)
	{
		for (uint32_t i = range.begin(); i < range.end(); ++i)
		{
			// upsampled information
			igl::writeOBJ(outputFolderWrinkles + "wrinkledMesh_" + std::to_string(i) + ".obj", wrinkledVList[i], upsampledTriF);
			saveAmp4Render(ampFieldsList[i], outputFolderAmp + "upAmp_" + std::to_string(i) + ".cvs", globalAmpMin, globalAmpMax);
			savePhi4Render(phaseFieldsList[i], outputFolderPhase + "upPhase" + std::to_string(i) + ".cvs");
		}
	};

	tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nframes);
	tbb::parallel_for(rangex, savePerFrame);

	return true;
}

double computeEnergyAndGradient(
    const Eigen::VectorXcd& psi,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& face_grad_phis_2d,
    Eigen::VectorXcd& grad) 
{
    int num_vertices = psi.size();
    int num_faces = F.rows();
    
    grad.setZero(num_vertices);
    double total_energy = 0.0;

    for (int f_idx = 0; f_idx < num_faces; ++f_idx) {
        int i = F(f_idx, 0);
        int j = F(f_idx, 1);
        int k = F(f_idx, 2);

        // Get the face-local 2D omega components
        double c1 = face_grad_phis_2d(f_idx, 0); // component 1
        double c2 = face_grad_phis_2d(f_idx, 1); // component 2
        
        // Edge omegas (w) derived from face components
        double w_ij = c1;
        double w_ki = -c2;
        double w_jk = c2 - c1;

        // --- Forward direction (using w_ij, w_jk, w_ki) ---
        Complex exp_ij_fwd = std::exp(Complex(0.0, w_ij));
        Complex exp_jk_fwd = std::exp(Complex(0.0, w_jk));
        Complex exp_ki_fwd = std::exp(Complex(0.0, w_ki));
        
        // Residuals: r_uv = psi[v] - psi[u] * exp(i * w_uv)
        Complex r_ij = psi(j) - psi(i) * exp_ij_fwd;
        Complex r_jk = psi(k) - psi(j) * exp_jk_fwd;
        Complex r_ki = psi(i) - psi(k) * exp_ki_fwd;
        
        double E_fwd = std::norm(r_ij) + std::norm(r_jk) + std::norm(r_ki);
        
        // --- Reverse direction (using -w) ---
        // exp(-i * w) = std::exp(Complex(0.0, -w)) = 1.0 / exp(i * w)
        Complex exp_ij_rev = std::conj(exp_ij_fwd);
        Complex exp_jk_rev = std::conj(exp_jk_fwd);
        Complex exp_ki_rev = std::conj(exp_ki_fwd);

        Complex nr_ij = psi(j) - psi(i) * exp_ij_rev;
        Complex nr_jk = psi(k) - psi(j) * exp_jk_rev;
        Complex nr_ki = psi(i) - psi(k) * exp_ki_rev;

        double E_rev = std::norm(nr_ij) + std::norm(nr_jk) + std::norm(nr_ki);
        
        total_energy += 0.5 * (E_fwd + E_rev);

        // --- Gradient Calculation ---
        // Gradient for E_fwd:
        grad(i) += (r_ki * std::conj(exp_ki_fwd) - r_ij * exp_ij_fwd); // Note: Python conjugation is handled implicitly
        grad(j) += (r_ij * std::conj(exp_ij_fwd) - r_jk * exp_jk_fwd);
        grad(k) += (r_jk * std::conj(exp_jk_fwd) - r_ki * exp_ki_fwd);

        // Gradient for E_rev:
        grad(i) += (nr_ki * std::conj(exp_ki_rev) - nr_ij * exp_ij_rev);
        grad(j) += (nr_ij * std::conj(exp_ij_rev) - nr_jk * exp_jk_rev);
        grad(k) += (nr_jk * std::conj(exp_jk_rev) - nr_ki * exp_ki_rev);
    }
    
    return total_energy;
}

void optimizePsi(
    Eigen::VectorXcd& psi,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& face_grad_phis_2d,
    double lr, 
    int max_iters, 
    double tol) 
{
    double prev_energy = -1.0;
    Eigen::VectorXcd grad;

    for (int it = 0; it < max_iters; ++it) {
        double energy = computeEnergyAndGradient(psi, F, face_grad_phis_2d, grad);

        // Update psi: psi -= lr * grad
        psi -= lr * grad;

        // Projection: psi /= |psi| (Projection back to unit magnitude)
        // Eigen's cwiseAbs() computes the magnitude of each complex number
        psi = psi.cwiseQuotient(psi.cwiseAbs());

        // Convergence check (energy change)
        if (prev_energy > 0.0 && std::abs(prev_energy - energy) < tol) {
            std::cout << "Converged at iter " << it << ", energy = " << energy << std::endl;
            break;
        }

        prev_energy = energy;
        // Optionally print energy every 100 iterations
        if (it % 100 == 0) {
            std::cout << "Iteration " << it << ", Energy = " << energy << std::endl;
        }
    }
}

void optimizeAndSaveZvals()
{
    if (numFrames == 0) return;
    if (faceOmegaList2Dinput.size() < numFrames) {
        std::cerr << "Error: Omega list size mismatch with numFrames." << std::endl;
        return;
    }

    // Resize the output container to hold all results
    zList.resize(numFrames);
    
    // Placeholder for the Z-values during optimization
    Eigen::VectorXcd psi_eigen; 

    // Define consistent optimization parameters
    const double learning_rate = 2e-3;
    const double tolerance = 1e-6;
    
    // Define Iteration Counts
    const int initial_iters = 2000; // For stabilization from random start (Frame 0)
    const int subsequent_iters = 20; // For smooth transition between frames (Frame i > 0)


    for (int i = 0; i < numFrames; ++i) 
    {
        // --- 1. Determine Initialization and Iteration Count ---
        
        int max_iters;
        if (i == 0) {
            // Frame 0: Start from random initialization, use high iterations
            max_iters = initial_iters;
            copyComplexStdVectorToEigen(initZvals, psi_eigen);
            
        } else {
            // Frames i > 0: Use the solution from the previous frame (i-1) as warm start
            max_iters = subsequent_iters;
            // Use the solution already stored in zList[i - 1]
            copyComplexStdVectorToEigen(zList[i - 1], psi_eigen); 
        }

        // --- 2. Get Input Omega Data for Current Frame ---
        const Eigen::MatrixXd& faceOmega2D = faceOmegaList2Dinput[i];

        // --- 3. Run Optimization ---
        std::cout << "\nOptimizing Frame " << i << " (" << max_iters << " iterations)..." << std::endl;

        optimizePsi(psi_eigen, triF, faceOmega2D, learning_rate, max_iters, tolerance);

        // --- 4. Save Result ---
        // Convert the optimized Eigen vector back to the application's std::vector format and store it.
        copyComplexEigenToStdVector(psi_eigen, zList[i]);
        
        std::cout << "Frame " << i << " Z-values successfully optimized and saved." << std::endl;
    }
}

int main(int argc, char** argv)
{
	CLI::App app("Wrinkle Interpolation");
	app.add_option("input,-i,--input", args.input, "Input model")->required()->check(CLI::ExistingFile);
	app.add_option("-g,--gradTol", args.gradTol, "The gradient tolerance for optimization.");
	app.add_option("-x,--xTol", args.xTol, "The variable update tolerance for optimization.");
	app.add_option("-f,--fTol", args.fTol, "The functio value update tolerance for optimization.");
	app.add_option("-n,--numIter", args.numIter, "The number of iteration for optimization.");
	app.add_option("-a,--ampScaling", args.ampScale, "The amplitude scaling for wrinkled surface upsampling.");
	app.add_flag("-r,--reoptimize", args.reOptimize, "Whether to reoptimize the input, default is false");

	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError& e) {
		return app.exit(e);
	}

	if (!loadSolvedProblem())
	{
		std::cout << "failed to load file." << std::endl;
		return 1;
	}

	// if (args.reOptimize)
	// {
	// 	// solve for the path from source to target
	// 	solveKeyFrames(initZvals, initOmega, faceFlags, omegaList, zList);
	// }

	//need to read from files, i get gradphi and wrinkle amplitudes
	//need to convert that to zvals and omega for every frame
	//updateeverything will handle the upsampling
	//need to update the mesh at each point of time too!!! this is over a static mesh

	//initialise zvals as unit valued complex numbers (random)
	initializeRandomUnitNormZvals(triV[0]);

	//go through the optimization process
	//need to populate zlist
	optimizeAndSaveZvals();

	// this function requires the upsampled mesh.  (only one frame)
	// but at the same time, it requires a list of zvals and omegas for wrinkle propagation across frames
	updateEverythingForSaving();
	
	saveProblem();
	saveForRender();

	return 0;
}