extern "C"

#include <stdio.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
//#include <sys/resource.h>
//#include "cuPrintf.cu"

// includes, project
//#include <cutil_inline.h>



//**************************************************************************
//						Defining the Constants and global
//**************************************************************************
#define MAPFILE "Test4"
#define NUM_AGENTS 45
int numAgents = 0;



//**************************************************************************
//						Device Structs
//**************************************************************************

__device__ struct AStarNode{
	int2 p;
	int2 father;
	int F;
	int G;
	int H;
};

__device__ struct passByReference{
	struct AStarNode* list;
	int begin;
	int end;
};


//**************************************************************************
//						Device Varaibles
//**************************************************************************
__device__ int* d_agentsX;
__device__ int* d_agentsY;
__device__ int* d_map;
__device__ int* d_checkPoint;
__device__ int* d_dim;
__device__ int* d_numAgents;
__device__ int2* d_walkThru;
__device__ int* d_tamWalk;
__device__ AStarNode* d_closedList;
__device__ AStarNode* d_openList;
__device__ int* d_closedListIndex;
__device__ int* d_openListBegin;
__device__ int* d_openListEnd;



//**************************************************************************
//						Host Structs
//**************************************************************************

struct point{
	int x;
	int y;
}typedef Point;


struct list{
	Point *p;
	struct list *prox;
}typedef List;


//**************************************************************************
//						Host Varaibles
//**************************************************************************
int* h_agentsX;
int* h_agentsY;
int* h_map;
int* h_checkPoint;
int* h_dim;
int* h_numAgents;
int2* h_walkThru;
int* h_tamWalk;


//**************************************************************************
//						Cuda Functions
//**************************************************************************
int InitCUDA();
__global__ void kernel(int exec, int* agentsX,int* agentsY, int* nAgents, int* map, int* dim, int* checkPoint, int2* walkThru, int* tamWalk, struct AStarNode* closedList, struct AStarNode* openList, int* closedListIndex, int* openListBegin, int* openListEnd);
__device__ struct passByReference addAdjacents(struct AStarNode root, int index, int* map, int dim, int jump, struct AStarNode* openList, int begin, int end, int2 endPoint);
__device__ void initConstants();
__device__ struct passByReference inTheList(struct AStarNode a, int index, int jump, struct AStarNode* openList, int begin, int end);
__device__ int calculateH(int2 P, int2 To);
__device__ struct AStarNode* sort(struct AStarNode* list, int index, int jump, int begin, int end);
__device__ int pointInside(struct AStarNode* list, int tam, int2 p, int index, int jump);

//**************************************************************************
//						Host Functions
//**************************************************************************

//Copy Funcions
void copyAgentVectors(Point **agents);
void copyMapMatrix(int **map,int dim);
void copycheckPoint(Point* checkPoint);
void copyVariables(int dim, int numAgents);
void copyWalkThru(int dim, int numAgents);


//Get Funcions

Point **getAgentVectors(Point **agents);
int **getMapMatrix(int** sceneMap, int dim);
void getWalkThru(int dim, int numAgents);

//Free functions
void freeAgentVectors(Point **agents);
void freeMapMatrix(int **map,int dim);
void freecheckPoint(Point* checkPoint);
void freeVariables();
void freeWalkThru();



//Matrix Stuff
int **getMap(char fileName[], int *dim, Point** checkPoint, List** aS);

//List Stuff
List* initList();
List* insertList(List *l, Point *p);
List* removeList(List *l, Point *p);
int lengthList(List *l);
void freeList(List *l);

//Agent Stuff
Point** initAgents(int maxAgents, int numberOfAgents ,List* aS);

//Auxiliar Functions
void printMat(int **mat, int dim);
void printAgents(Point** agents);


//**************************************************************************
//						Main
//**************************************************************************

int main(void){
	//Var Declarations
	int dim;
	List* agentSpots = initList();			//Available Agent Spots
	int **sceneMap = NULL;					//Map of this Scene
	Point *checkPoint = NULL;				//Agents Checkpoint
	Point **agents = NULL;
	int elapTicks;
	double elapMilli;
	clock_t begin,end;
	
	//Init Map
	sceneMap = getMap(MAPFILE, &dim, &checkPoint, &agentSpots);
	
	//Init Agents
	agents = initAgents(lengthList(agentSpots), NUM_AGENTS, agentSpots);
	
	
	if(InitCUDA()){
		//Preparing the Variables to the Kernel	
		copyAgentVectors(agents);
	
		copyMapMatrix(sceneMap,dim);
	
		copycheckPoint(checkPoint);
	
		copyVariables(dim, numAgents);
		
		copyWalkThru(dim, numAgents);
		
		begin = clock();
		
//		cudaPrintfInit();
		//Calling the Kernel
		kernel<<<1,numAgents>>>(1, d_agentsX,d_agentsY, d_numAgents, d_map, d_dim, d_checkPoint, d_walkThru, d_tamWalk, d_closedList, d_openList, d_closedListIndex, d_openListBegin, d_openListEnd);
		//cudaThreadSynchronize();
		//cudaPrintfDisplay(stdout, true);
		//Gettin the walkthruback
		//getWalkThru(dim, numAgents);
						
		kernel<<<1,numAgents>>>(2, d_agentsX,d_agentsY, d_numAgents, d_map, d_dim, d_checkPoint, d_walkThru, d_tamWalk, d_closedList, d_openList, d_closedListIndex, d_openListBegin, d_openListEnd);
//		cudaPrintfDisplay(stdout, true);
		cudaThreadSynchronize();
		
		end = clock();
		
		elapTicks = end - begin;        //the number of ticks from Begin to End
		elapMilli = elapTicks/(double)1000;     //milliseconds from Begin to End
		
		printf("Tempo de execucao = %f\n",elapMilli);
		
//		cudaPrintfEnd();
		//Getting everything back
		agents = getAgentVectors(agents);
		sceneMap = getMapMatrix(sceneMap,dim);
		
		//Poor debug system xD
//		printAgents(agents);
//		printMat(sceneMap,dim);
		
		
		//Free everything
		freeAgentVectors(agents);
		freeMapMatrix(sceneMap,dim);
		freecheckPoint(checkPoint);
		freeVariables();
	}
	
}

//**************************************************************************
//						Cuda Functions
//**************************************************************************

__global__ void kernel(int exec, int* agentsX,int* agentsY, int* nAgents, int* map, int* dim, int* checkPoint, int2* walkThru, int* tamWalk, struct AStarNode* closedList, struct AStarNode* openList, int* closedListIndex, int* openListBegin, int* openListEnd){
	//Init all the general stuff
	int testes = 0;
	int myIndex = threadIdx.x;
	int i;
	int2 myPoint; 
	int2 endPoint;
	int dimMatrix = *dim;
	int numAgents = *nAgents;
	int jump = dimMatrix*dimMatrix;
	
	myPoint.x = agentsX[myIndex];
	myPoint.y = agentsY[myIndex];
	
	endPoint.x = checkPoint[0];
	endPoint.y = checkPoint[1];
	
	
	
	//Sync the threads before start any of the executions
	__syncthreads();
	
	
	//If to decide which execution will do.
	if(exec == 1){
		//Calculate the A* Algorithm
		struct passByReference respFunc;
		
		openListBegin[myIndex] = 0;
		openListEnd[myIndex] = 0;
		closedListIndex[myIndex] = 0;
		tamWalk[myIndex] = 0;
		
		
		struct AStarNode root;
		root.p.x = agentsX[myIndex];
		root.p.y = agentsY[myIndex];
		root.father.x = -1;
		root.father.y = -1;
		root.F = 0;
		root.G = 0;
		root.H = 0;
		
		respFunc = addAdjacents(root, myIndex, map, dimMatrix, jump, openList, openListBegin[myIndex], openListEnd[myIndex], endPoint);
		openList = respFunc.list;
		openListBegin[myIndex] = respFunc.begin;
		openListEnd[myIndex] = respFunc.end;
		
		
		
		openList = sort(openList, myIndex, jump, openListBegin[myIndex], openListEnd[myIndex]);
		
		
		closedList[(myIndex*jump) + closedListIndex[myIndex]] = root;
		closedListIndex[myIndex]++;
		/*
		if(1){
			int i;
			agentsX[myIndex] = respFunc.begin;
			agentsY[myIndex] = respFunc.end;
			cuPrintf("openListBegin[myIndex] = %d , openListEnd[myIndex] = %d\n", openListBegin[myIndex], openListEnd[myIndex]);
			cuPrintf("Pontos:\n");
			for(i=0 ; i<respFunc.end ; i++){
				cuPrintf("[%d,%d]\n", openList[i].p.x, openList[i].p.y);
			}
			
			return;
		}
		*/
		
		//Loop of the algorithm
		while((openListEnd[myIndex] - openListBegin[myIndex]) > 0){
			struct AStarNode A = openList[(myIndex*jump) + openListBegin[myIndex]];
			
			
			
			closedList[(myIndex*jump) + closedListIndex[myIndex]] = A;
			closedListIndex[myIndex]++;
			
			//Another condition to go out
			if(pointInside(closedList, closedListIndex[myIndex], endPoint, myIndex, jump)){
//				cuPrintf("ACHEI O 0,0!!! /o/\n");
				break;
			}
			
			
			openListBegin[myIndex]++;
			
			respFunc = addAdjacents(A, myIndex, map, dimMatrix, jump, openList, openListBegin[myIndex], openListEnd[myIndex], endPoint);
			openList = respFunc.list;
			openListBegin[myIndex] = respFunc.begin;
			openListEnd[myIndex] = respFunc.end;
			
			openList = sort(openList, myIndex, jump, openListBegin[myIndex], openListEnd[myIndex]);
			
//			if(1){
//				int i;
//				agentsX[myIndex] = respFunc.begin;
//				agentsY[myIndex] = respFunc.end;
//				cuPrintf("openListBegin[myIndex] = %d , openListEnd[myIndex] = %d\n", openListBegin[myIndex], openListEnd[myIndex]);
//				cuPrintf("Open List: Pontos:\n");
//				for(i=openListBegin[myIndex] ; i<openListEnd[myIndex] ; i++){
//					cuPrintf("[%d,%d] , F = %d\n", openList[(myIndex*jump) + i].p.x, openList[(myIndex*jump) + i].p.y, openList[(myIndex*jump) +i].F);
//				}
			
//				cuPrintf("closedListIndex[myIndex] = %d\n", closedListIndex[myIndex]);
//				cuPrintf("ClosedList: Pontos:\n");
//				for(i=0 ; i<closedListIndex[myIndex] ; i++){
//					cuPrintf("[%d,%d]\n", closedList[(myIndex*jump) + i].p.x, closedList[(myIndex*jump) + i].p.y);
//				}
//				testes++;
				//if(testes > 4) return;
//			}
			
		}
		
		if(openListEnd[myIndex] - openListBegin[myIndex] == 0){	
//			cuPrintf("ERROR!\n");
			return;	//error!
		}	
		
		//Finish Him!
		struct AStarNode k = closedList[(myIndex*jump) + (closedListIndex[myIndex]-1)];
//		cuPrintf("Vamo fechar a bagaça!\n");
//		cuPrintf("k = [%d,%d], pai = [%d,%d]\n", k.p.x, k.p.y, k.father.x, k.father.y);
		walkThru[(myIndex*jump) + tamWalk[myIndex]] = k.p;
		tamWalk[myIndex]++;
		while((k.p.x != myPoint.x) && (k.p.y != myPoint.y)){
			int i=0;
			while((closedList[(myIndex*jump) + i].p.x != k.father.x && closedList[(myIndex*jump) + i].p.y != k.father.y)){
				i++;
				
//				cuPrintf("i>=jump? %d\n",i>=jump);
				if(i>=jump){
					//agentsY[myIndex] = -7;
//					cuPrintf("Tá dando erro ¬¬ \n");
					return;		//error!
				}
				
			}
			
			walkThru[(myIndex*jump) + tamWalk[myIndex]] = closedList[(myIndex*jump) + i].p;
			tamWalk[myIndex]++;
			k = closedList[(myIndex*jump) + i];
		}
		
//		cuPrintf("Walkbefore:\n");
//		for(i=0 ; i< (tamWalk[myIndex]) ; i++){
//			cuPrintf("[%d,%d]\n",walkThru[(myIndex*jump) + i].x,walkThru[(myIndex*jump) + i].y);
//		}
		//walkThru[(myIndex*jump) + tamWalk[myIndex]] = k.p;
		//tamWalk[myIndex]++;
//		cuPrintf("tamWalk[myIndex] = %d\n", tamWalk[myIndex]);
		//invert the vector
//		cuPrintf("Invertendo!\n");
		for(i=0 ; i< (tamWalk[myIndex]/2) ; i++){
			int2 aux = walkThru[(myIndex*jump) + i];
			walkThru[(myIndex*jump) + i] = walkThru[(myIndex*jump) + ((tamWalk[myIndex]-i)-1)];
			walkThru[(myIndex*jump) + ((tamWalk[myIndex]-i)-1)] = aux;
		}
		
		//Debug
//		cuPrintf("Walk:\n");
//		for(i=0 ; i< (tamWalk[myIndex]) ; i++){
//			cuPrintf("[%d,%d]\n",walkThru[(myIndex*jump) + i].x,walkThru[(myIndex*jump) + i].y);
//		}
		
	}
	else if(exec == 2){
	//Walk with the agents
	int i=0;
	int2 myActualPos;
	myActualPos.x = -1;
	myActualPos.y = -1;
//	cuPrintf("LETS WALK YOUR LAZY AGENT!\n");
	while(i<tamWalk[myIndex]){
		//cuPrintf("Agent %d\n",myIndex);
		
		//if(map[newPos.x*dimMatrix) + newPos.y != 1]){
			//if(myActualPos.x != -1 && myActualPos.y != -1){
				map[(myActualPos.x*dimMatrix) + myActualPos.y] = 0;
			//}
			myActualPos = walkThru[(myIndex*jump) + i];
			map[(myActualPos.x*dimMatrix) + myActualPos.y] = 1;
//			cuPrintf("Agent %d andou pra [%d,%d]\n",myIndex, myActualPos.x, myActualPos.y);
			i++;
		//}
	}
	map[(endPoint.x*dimMatrix) + endPoint.y] = 2;
	
	
	}
	else{
		return;
	}
	
	
}

__device__ int pointInside(struct AStarNode* list, int tam, int2 p, int index, int jump){
	int i;
	for(i=0; i<tam ; i++){
		struct AStarNode A = list[(index*jump)+i];
		if((A.p.x == p.x) && (A.p.y == p.y)){
			return 1;
		}
	}
	return 0;
}

//Adding the Adjecents to the closedList
struct passByReference addAdjacents(struct AStarNode root, int index, int* map, int dim, int jump, struct AStarNode* openList, int begin, int end, int2 endPoint){
	int2 temp;
	struct passByReference resp;
	struct passByReference listBack;
	
	
	//North
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.y += 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);
	
	
	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 0){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = make_int2(root.p.x, root.p.y);
				a.G = 10;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}
	
	//South
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.y -= 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);
	
	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 10;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}

	//East
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.x += 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);

	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 10;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}
	
	//West
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.x -= 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);

	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 10;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}

	
	//Northeast
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.x += 1;
	temp.y += 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);

	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 14;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}
	
	//Northwest
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.x -= 1;
	temp.y += 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);

	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 14;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}
		
	//Southeast
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.x += 1;
	temp.y -= 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);

	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 14;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}
	
	//Southwest
	temp.x = root.p.x;
	temp.y = root.p.y;
	
	temp.x -= 1;
	temp.y -= 1;
//	cuPrintf("temp = [%d,%d]\n",temp.x, temp.y);

	//Test Conditions
	if(temp.x > (-1) && temp.x < dim){			//Test x dim
		if(temp.y > (-1) && temp.y < dim){		//Test y dim
			if(map[(temp.x*dim) + temp.y] != 1){	//Testing if the position is an obstacle
				struct AStarNode a;
				a.p = temp;
				a.father = root.p;
				a.G = 14;
				a.H = calculateH(temp,endPoint);
				a.F = a.G + a.H;
				listBack = inTheList(a, index, jump, openList, begin, end);
				openList = listBack.list;
				
				if(!listBack.begin){
					openList[index*jump + end] = a;
					end++;
				}
			}
		}
	}
	resp.list = openList;
	resp.begin = begin;
	resp.end = end;
	return resp;
	
	
	
}

//Sort the elements in the list in a range.
__device__ struct AStarNode* sort(struct AStarNode* list, int index, int jump, int begin, int end){
	int i,stop, times = 0;
	stop=1;
	while(stop){
		stop=0;
		for(i=begin; i<(end-1) ; i++){
			if(list[index*jump + i].F > list[index*jump + (i+1)].F){
				times++;
				stop = 1;
				struct AStarNode aux = list[index*jump + i];
				list[index*jump + i] = list[index*jump + (i+1)];
				list[index*jump + (i+1)] = aux;
			}
		}
	}
//	cuPrintf("Sort changes %d times \n",times);
	return list;
}

//See if theres a element A in the open list
__device__ struct passByReference inTheList(struct AStarNode a, int index, int jump, struct AStarNode* openList, int begin, int end){
	int i;
	struct passByReference resp;
	
	for(i = begin; i<end ; i++){
		struct AStarNode p = openList[index*jump + i];
		if((p.p.x == a.p.x) && (p.p.y == a.p.y)){
			if(p.G > a.G){
				openList[index*jump + i] = a;
			}
			resp.list = openList;
			resp.begin = 1;
			return resp;
		}
	}
	resp.list = openList;
	resp.begin = 0;
	return resp;
}


__device__ int calculateH(int2 P, int2 To){

	int t1 = P.x - To.x;
	int t2 = P.y - To.y;

	if(t1 < 0) t1 *= -1;
	if(t2 < 0) t2 *= -1;
//	cuPrintf("P.x = %d , P.y = %d , To.x = %d , To.y = %d , t1 = %d , t2 = %d , t1+t2 = %d\n",P.x, P.y, To.x, To.y, t1, t2, t1+t2);
	return t1+t2;
}



int InitCUDA()
{
    int count;
	
    cudaGetDeviceCount(&count);
    if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return 0;
    }
	
    int i;
    for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
    }
	
    if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
    }
	
	cudaSetDevice(i);
	
	return 1;
}




//**************************************************************************
//						Other Functions
//**************************************************************************

//Print the agent vector
void printAgents(Point **agents){
	int i;
	printf("\nNúmero de Agentes: %d",numAgents);
	printf("\nAgents:\n");
	for(i=0 ; i<numAgents ; i++){
		printf("[%d,%d]\n", agents[i]->x,agents[i]->y);
	}
	printf("End of the Agents\n");
}

//Print the Matrix
void printMat(int **mat, int dim){
	int i,j;
	for(i=0 ; i<dim ; i++){
		for(j=0 ; j<dim ; j++){
			printf("%d  ",mat[i][j]);
		}
		printf("\n");
	}
	
}


//Get Back The Agent Vector
Point **getAgentVectors(Point **agents){
	int i;
	
	//Copying back from the Device
	cudaMemcpy(h_agentsX, d_agentsX, sizeof(int)*numAgents, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_agentsY, d_agentsY, sizeof(int)*numAgents, cudaMemcpyDeviceToHost);
	

	for(i=0 ; i<numAgents ; i++){
		
		agents[i]->x = h_agentsX[i];
		agents[i]->y = h_agentsY[i];
	}

	return agents;
}

//Get Back the Map Matrix
int **getMapMatrix(int** sceneMap, int dim){
	int i,j;
	
	//Copying back from the Device
	cudaMemcpy(h_map, d_map, sizeof(int)*dim*dim, cudaMemcpyDeviceToHost);
	for(i=0 ; i<dim ; i++){
		
		for(j=0 ; j<dim ; j++){
			 sceneMap[i][j] = h_map[(i*dim) + j];
		}
	}
	
	return sceneMap;
}

//Getting back the Walkthru Stuff
void getWalkThru(int dim, int numAgents){
	int tam = sizeof(int2) * numAgents * (dim*dim);
	int i,j;

	//Copying back from the Device
	cudaMemcpy(h_walkThru, d_walkThru, tam, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tamWalk, d_tamWalk, sizeof(int)*numAgents, cudaMemcpyDeviceToHost);
	
	//Kill all the unecessary stuff
	cudaFree(d_openList);
	cudaFree(d_closedList);
	cudaFree(d_closedListIndex);
	cudaFree(d_openListBegin);
	cudaFree(d_openListEnd);
	
	//Debug System.
	for(i = 0; i<numAgents ; i++){
		printf("\nAgent %d walk, tamWalk = %d\n",i, h_tamWalk[i]);
		for(j=0 ; j<h_tamWalk[i] ; j++){
			printf("[%d,%d];\n",h_walkThru[i * (dim*dim) + j].x, h_walkThru[i * (dim*dim) + j].y);
		}
	}
	
	
}

//Free the walkthru stuff

void freeWalkThru(){
	free(h_tamWalk);
	free(h_walkThru);
	
	cudaFree(d_tamWalk);
	cudaFree(d_walkThru);
}

//Free functions
void freeAgentVectors(Point **agents){
	int i;
	
	//Free the Device
	cudaFree(d_agentsX);
	cudaFree(d_agentsY);
	
	//Free the Host
	for(i=0 ; i<numAgents ; i++){
		free(agents[i]);
	}
	free(h_agentsX);
	free(h_agentsY);
	free(agents);
}

void freeMapMatrix(int **map,int dim){
	int i;
	
	for(i=0 ; i<dim ; i++){
		
		//Free the Host
		
		free(map[i]);
	}
	//Free the Device
	cudaFree(d_map);
		
	//Free the Host
	free(h_map);
	free(map);
	
	
}

void freecheckPoint(Point* checkPoint){
	//Free the device
	cudaFree(d_checkPoint);
	
	//Free the Host
	free(h_checkPoint);
	free(checkPoint);
}

void freeVariables(){
	//Free the device
	cudaFree(d_dim);
	cudaFree(d_numAgents);
	
	//Free the host
	free(h_dim);
	free(h_numAgents);
}

//Free the walkthru stuff



//Copy the walkthru stuff
void copyWalkThru(int dim, int numAgents){
	//float tam = sizeof(int2) * numAgents * (dim*dim);
	int i;

	h_walkThru = (int2*) malloc (sizeof(int2) * numAgents * (dim*dim));
	
	h_tamWalk = (int*) malloc (sizeof(int)*numAgents);
	for(i=0 ; i<sizeof(int)*numAgents ; i++){
		h_tamWalk[i] = 0;
	}
	
	//Malloc on the device
	cudaMalloc((void**) &d_walkThru, sizeof(int2) * numAgents * (dim*dim));
	cudaMalloc((void**) &d_tamWalk, sizeof(int)*numAgents);
	cudaMalloc((void**) &d_closedList, sizeof(struct AStarNode) * (dim*dim) * numAgents);
	cudaMalloc((void**) &d_openList, sizeof(struct AStarNode) * (dim*dim) * numAgents);
	cudaMalloc((void**) &d_closedListIndex, sizeof(int) * (numAgents));
	cudaMalloc((void**) &d_openListBegin, sizeof(int) * (numAgents));
	cudaMalloc((void**) &d_openListEnd, sizeof(int) * (numAgents));
	
}

//Coping the Variables
void copyVariables(int dim, int numAgents){
	h_dim = (int*) malloc (sizeof(int));
	h_numAgents = (int*) malloc (sizeof(int));
	
	*h_dim = dim;
	*h_numAgents = numAgents;
	
	//Malloc on the device
	cudaMalloc((void**) &d_dim, sizeof(int));
	cudaMalloc((void**) &d_numAgents, sizeof(int));
	
	//Copying to the device
	cudaMemcpy(d_dim, h_dim, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_numAgents, h_numAgents, sizeof(int), cudaMemcpyHostToDevice);
	
	
}

//Coping the checkPoint
void copycheckPoint(Point* checkPoint){
	h_checkPoint = (int*) malloc (sizeof(int)*2);
	h_checkPoint[0] = checkPoint->x;
	h_checkPoint[1] = checkPoint->y;

	//Malloc on the device
	cudaMalloc((void**) &d_checkPoint, (sizeof(int)*2));
	
	//Copying to the device
	cudaMemcpy(d_checkPoint, h_checkPoint, (sizeof(int)*2), cudaMemcpyHostToDevice);
	
}

//Coping the map

void copyMapMatrix(int **map, int dim){
	int i,j;
	h_map = (int*) malloc (sizeof(int)*dim*dim);
	for(i=0 ; i<dim ; i++){
		for(j=0 ; j<dim ; j++){
			h_map[(i*dim) + j] = map[i][j];
		}
	}
	
	//Malloc on the device
	cudaMalloc((void**) &d_map, (sizeof(int)*dim*dim));
	
	//Copying to the device
	cudaMemcpy(d_map, h_map, (sizeof(int)*dim*dim), cudaMemcpyHostToDevice);
	
}



//Coping the vectors

void copyAgentVectors(Point **agents){
	int i;
	d_agentsX = 0;
	h_agentsX = (int*) malloc (sizeof(int)*numAgents);
	h_agentsY = (int*) malloc (sizeof(int)*numAgents);
	
	for(i=0 ; i<numAgents ; i++){
		h_agentsX[i] = agents[i]->x;
		h_agentsY[i] = agents[i]->y;
	}
	
	//Malloc on the device
	cudaMalloc((void**) &d_agentsX, sizeof(int)*numAgents);
	cudaMalloc((void**) &d_agentsY, sizeof(int)*numAgents);
	
	if(d_agentsX == 0) printf("OH FUCK!");
	
	//Copying to the device
	cudaMemcpy(d_agentsX, h_agentsX, sizeof(int)*numAgents, cudaMemcpyHostToDevice);
	cudaMemcpy(d_agentsY, h_agentsY, sizeof(int)*numAgents, cudaMemcpyHostToDevice);
}




//Getting the Map
int **getMap(char fileName[], int *dim, Point** checkPoint, List** aS){
	FILE* fp;
	int i,j, **map;
	
	fp = fopen(fileName, "rt");
	if(!fp){
		printf("\nArquivo não encontrado! Finalizando o programa!\n");
		exit(1);
	}
	
	fscanf(fp, " %d", dim);
	
	map = (int**) malloc (sizeof(int*)*(*dim));
	for(i=0 ; i<(*dim) ; i++){
		map[i] = (int*) malloc (sizeof(int)*(*dim));
		for(j=0 ; j<(*dim) ; j++){
			fscanf(fp, " %d", &map[i][j]);
			if(map[i][j] == 2){
				Point *p = (Point*) malloc (sizeof(Point));
				p->x = i;
				p->y = j;
				*checkPoint = p;
			}
			else if(map[i][j] == 3){
				Point *p = (Point*) malloc (sizeof(Point));
				p->x = i;
				p->y = j;
				(*aS) = insertList((*aS), p);
				map[i][j] = 0;
			}
		}
	}
	fclose(fp);
	return map;
}



//Init the List
List* initList(){
	return NULL;
}

//Insert things on the List
List* insertList(List *l, Point *p){
	List *n = (List*) malloc (sizeof(List));
	n -> p = p;
	n -> prox = NULL;
	
	if(!l)
		return n;
	else{
		n -> prox = l;
		return n;
	}
	
}

//Remove Things on the List
List* removeList(List *l, Point *p){
	List *ant, *next;
	if(!l || !p)
		return l;
	
	
	ant = NULL;
	next = l;
	
	while(next || ((p->x != next->p->x) && (p->y != next->p->y))){
		ant = next;
		next = next->prox;
	}
	
	if(!ant){
		l = l->prox;
		free(next);
	}
	else if(next){		
		ant -> prox = next ->prox;
		free(next);
		
	}
	return l;
	
}

//Length of the list
int lengthList(List *l){
	int i=0;
	
	if(l){
		List *aux = l;
		while(aux){
			i++;
			aux = aux->prox;
		}
		return i;
	}
	else{
		return i;
	}
}

//Free the List
void freeList(List *l){
	List *aux=l;
	while(aux){
		l = l->prox;
		free(aux -> p);
		free(aux);
		aux = l;
	
	}
	
}

//Init the agents
Point** initAgents(int maxAgents, int numberOfAgents ,List* aS){
	int i;
	Point **agents;
	List* aux;
	
	if(maxAgents >= numberOfAgents){
		numAgents = numberOfAgents;
		
	}
	else{
		numAgents = maxAgents;
		
	}
	
	agents = (Point**) malloc (sizeof(Point*)*numAgents);

	aux = aS;
	for(i=0 ; i<numAgents ; i++){
		agents[i] = (Point*)malloc(sizeof(Point));
		
		agents[i] = aux->p;
		aux = aux->prox;
		
	}
	
	return agents;
}


