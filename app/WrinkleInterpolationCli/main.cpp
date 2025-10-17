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


struct PickedFace
{
	int fid = -1;
	double ampChangeRatio = 1.;
	int effectiveRadius = 5;
	int interfaceDilation = 5;
	VecMotionType freqVecMotion = Enlarge;
	double freqVecChangeValue = 1.;
	bool isFreqAmpCoupled = false;

	std::vector<int> effectiveFaces = {};
	std::vector<int> interFaces = {};
	std::vector<int> effectiveVerts = {};
	std::vector<int> interVerts = {};

	void buildEffectiveFaces(int nfaces)
	{
		effectiveFaces.clear();
		effectiveVerts.clear();
		interFaces.clear();
		interVerts.clear();

		if (fid == -1 || fid >= nfaces)
			return;
		else
		{
			Eigen::VectorXi curFaceFlags = Eigen::VectorXi::Zero(triF.rows());
			curFaceFlags(fid) = 1;
			Eigen::VectorXi curFaceFlagsNew = curFaceFlags;
			regEdt.faceDilation(curFaceFlagsNew, curFaceFlags, effectiveRadius);
			regEdt.faceDilation(curFaceFlags, curFaceFlagsNew, interfaceDilation);

			Eigen::VectorXi vertFlags, vertFlagsNew;

			faceFlags2VertFlags(triMesh, triV[0].rows(), curFaceFlags, vertFlags);
			faceFlags2VertFlags(triMesh, triV[0].rows(), curFaceFlagsNew, vertFlagsNew);

			for (int i = 0; i < curFaceFlags.rows(); i++)
			{
				if (curFaceFlags(i))
					effectiveFaces.push_back(i);
				else if (curFaceFlagsNew(i))
					interFaces.push_back(i);
			}


			for (int i = 0; i < vertFlags.rows(); i++)
			{
				if (vertFlags(i))
					effectiveVerts.push_back(i);
				else if (vertFlagsNew(i))
					interVerts.push_back(i);
			}
		}
	}
};

std::vector<PickedFace> pickFaces;

bool addSelectedFaces(const PickedFace face, Eigen::VectorXi& curFaceFlags, Eigen::VectorXi& curVertFlags)
{
	for (auto& f : face.effectiveFaces)
		if (curFaceFlags(f))
			return false;

	for (auto& v : face.effectiveVerts)
		if (curVertFlags(v))
			return false;

	for (auto& f : face.effectiveFaces)
		curFaceFlags(f) = 1;
	for (auto& v : face.effectiveVerts)
		curVertFlags(v) = 1;

	return true;
}


InitializationType initType = SeperateLinear;
double zuenkoTau = 0.1;
int zuenkoIter = 5;

static void buildEditModel(const Eigen::MatrixXd& pos, const MeshConnectivity& mesh, const std::vector<VertexOpInfo>& vertexOpts, const Eigen::VectorXi& faceFlag, int quadOrd, double spatialAmpRatio, double spatialEdgeRatio, double spatialKnoppelRatio, int effectivedistFactor, std::shared_ptr<IntrinsicFormula::WrinkleEditingModel>& editModel)
{
	editModel = std::make_shared<IntrinsicFormula::WrinkleEditingCWF>(pos, mesh, vertexOpts, faceFlag, quadOrd, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor);
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

bool loadProblem()
{
	std::string loadFileName = args.input;
	std::cout << "load file in: " << loadFileName << std::endl;
	using json = nlohmann::json;
	std::ifstream inputJson(loadFileName);
	if (!inputJson) {
		std::cerr << "missing json file in " << loadFileName << std::endl;
		return false;
	}

	std::string filePath = loadFileName;
	std::replace(filePath.begin(), filePath.end(), '\\', '/'); // handle the backslash issue for windows
	int id = filePath.rfind("/");
	workingFolder = filePath.substr(0, id + 1);
	std::cout << "working folder: " << workingFolder << std::endl;

	json jval;
	inputJson >> jval;

	std::string meshFile = jval["mesh_name"];
	upsampleTimes = jval["upsampled_times"];


	meshFile = workingFolder + meshFile;
	igl::readOBJ(meshFile, triV, triF);
	triMesh = MeshConnectivity(triF);
	initialization(triV, triF, upsampledTriV, upsampledTriF);
	

	quadOrder = jval["quad_order"];
	numFrames = jval["num_frame"];
    if (jval.contains(std::string_view{ "wrinkle_amp_ratio" }))
    {
        if(args.ampScale == 1)
            args.ampScale = jval["wrinkle_amp_ratio"];
    }

	isSelectAll = jval["region_global_details"]["select_all"];
	isCoupled = jval["region_global_details"]["amp_omega_coupling"];
	selectedMagValue = jval["region_global_details"]["amp_operation_value"];
	selectedMotionValue = jval["region_global_details"]["omega_operation_value"];
	std::string optype = jval["region_global_details"]["omega_operation_motion"];

	if (optype == "None")
		selectedMotion = None;
	else if (optype == "Enlarge")
		selectedMotion = Enlarge;
	else if (optype == "Rotate")
		selectedMotion = Rotate;
	else
		selectedMotion = None;

	pickFaces.clear();


	int nedges = triMesh.nEdges();
	int nverts = triV.rows();

	std::string initAmpPath = jval["init_amp"];
	std::string initOmegaPath = jval["init_omega"];
	std::string initZValsPath = "zvals.txt";
	if (jval.contains(std::string_view{ "init_zvals" }))
	{
		initZValsPath = jval["init_zvals"];
	}

	if (!loadEdgeOmega(workingFolder + initOmegaPath, nedges, initOmega)) {
		std::cout << "missing init edge omega file." << std::endl;
		return false;
	}

	if (!loadVertexZvals(workingFolder + initZValsPath, triV.rows(), initZvals))
	{
		std::cout << "missing init zval file, try to load amp file, and round zvals from amp and omega" << std::endl;
		if (!loadVertexAmp(workingFolder + initAmpPath, triV.rows(), initAmp))
		{
			std::cout << "missing init amp file: " << std::endl;
			return false;
		}

		else
		{
			Eigen::VectorXd edgeArea, vertArea;
			edgeArea = getEdgeArea(triV, triMesh);
			vertArea = getVertArea(triV, triMesh);
			IntrinsicFormula::roundZvalsFromEdgeOmegaVertexMag(triMesh, initOmega, initAmp, edgeArea, vertArea, triV.rows(), initZvals);
		}
	}
	else
	{
		initAmp.setZero(triV.rows());
		for (int i = 0; i < initZvals.size(); i++)
			initAmp(i) = std::abs(initZvals[i]);
	}

	std::string optZvals = jval["solution"]["opt_zvals"];
	std::string optOmega = jval["solution"]["opt_omega"];



	isLoadOpt = true;
	zList.clear();
	omegaList.clear();
	ampList.clear();
	for (int i = 0; i < numFrames; i++)
	{
		std::string zvalFile = workingFolder + optZvals + "/zvals_" + std::to_string(i) + ".txt";
		std::string edgeOmegaFile = workingFolder + optOmega + "/omega_" + std::to_string(i) + ".txt";
		
		std::vector<std::complex<double>> zvals;
		Eigen::VectorXd vertAmp;

		if (!loadVertexZvals(zvalFile, nverts, zvals))
		{
			isLoadOpt = false;
			break;
		}
		else {
			vertAmp.setZero(zvals.size());
			for (int i = 0; i < zvals.size(); i++) {
				vertAmp[i] = std::abs(zvals[i]);
			}
		}

		Eigen::VectorXd edgeOmega;
		if (!loadEdgeOmega(edgeOmegaFile, nedges, edgeOmega)) {
			isLoadOpt = false;
			break;
		}

		zList.push_back(zvals);
		omegaList.push_back(edgeOmega);
		ampList.push_back(vertAmp);
	}

	if (isLoadOpt)
	{
		std::cout << "load zvals and omegas from file!" << std::endl;
	}
	if (!isLoadOpt)
	{
		buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);

        if(!isLoadTar)
        {
            editModel->editCWFBasedOnVertOp(initZvals, initOmega, tarZvals, tarOmega);
        }
		editModel->initialization(initZvals, initOmega, tarZvals, tarOmega, numFrames - 2, true);

		zList = editModel->getVertValsList();
		omegaList = editModel->getWList();
		ampList = editModel->getRefAmpList();

	}
	else
	{
		buildEditModel(triV, triMesh, vertOpts, faceFlags, quadOrder, spatialAmpRatio, spatialEdgeRatio, spatialKnoppelRatio, effectivedistFactor, editModel);
		editModel->initialization(zList, omegaList, ampList, omegaList);
	}
	

	return true;
}

bool readFaces(std::string &facesFile, Eigen::MatrixXi &triF) 
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

bool loadSolvedProblem() {

	std::string verticesFile = "vertices.csv";
	std::string facesFile = "faces.csv";
	std::string amplitudesFile = "amplitudes.csv";

	facesFile = workingFolder + facesFile;
	readFaces(facesFile, triF);

	//will need to resize triV and ampList

	for(int i=0; i<numFrames; i++) {
		std::string curVerticesFile = workingFolder + verticesFile + '.' + std::to_string(i);
		readVertices(curVerticesFile, triV[i]);
		
		triMesh = MeshConnectivity(triF);
		initialization(triV[i], triF, upsampledTriV[i], upsampledTriF);

		std::string curAmplitudesFile = workingFolder + amplitudesFile + '.' + std::to_string(i);
		loadVertexAmp(curAmplitudesFile, triV[0].rows(), ampList[i]);

		//read face omega list

		//convert it to omegalist
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

	for (int i = 0; i < pickFaces.size(); i++)
	{
		curOpt = "None";
		if (pickFaces[i].freqVecMotion == Enlarge)
			curOpt = "Enlarge";
		else if (pickFaces[i].freqVecMotion == Rotate)
			curOpt = "Rotate";
		json pfJval =
		{
			{"face_id", pickFaces[i].fid},
			{"effective_radius", pickFaces[i].effectiveRadius},
			{"interface_dilation", pickFaces[i].interfaceDilation},
			{"omega_operation_motion", curOpt},
			{"omega_opereation_value", pickFaces[i].freqVecChangeValue},
			{"amp_operation_value", pickFaces[i].ampChangeRatio},
			{"amp_omega_coupling", pickFaces[i].isFreqAmpCoupled}
		};
		jval["region_local_details"].push_back(pfJval);
	}

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

	igl::writeOBJ(workingFolder + "mesh.obj", triV, triF);

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
	igl::writeOBJ(renderFolder + "basemesh.obj", triV, triF);
	igl::writeOBJ(renderFolder + "upmesh.obj", upsampledTriV, upsampledTriF);


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


	// this function requires the upsampled mesh.  (only one frame)
	// but at the same time, it requires a list of zvals and omegas for wrinkle propagation across frames
	updateEverythingForSaving();
	
	saveProblem();
	saveForRender();

	return 0;
}