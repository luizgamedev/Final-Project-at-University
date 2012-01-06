#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>

//Constantes Definidas
#define NUM_PACKS 10000
#define NUM_AGENTS 5000
//Variáveis e Funções do Kernel
void** d_agents;
void** d_packages;
void** d_times;	


__global__ void kernel(void** agents, int num_agents, void** packages, int num_packs/*, void** times*/);
__device__ void getPackage(void** packages, int pos);
__device__ void seePosition(int pos);
__device__ void allPackagesAreGone(void** packages, int num_packs, int* bye);

//Variáveis e Funções do Host
int InitCUDA();
void printVet(int* vet, int tam);
//void printVetFloat(clock_t* vet, int tam);


int main(void){
	int i;

	//Vetores Inicializando Vetores
	int* agents = (int*) malloc (sizeof(int) * NUM_AGENTS);
	int* packages = (int*) malloc (sizeof(int) * NUM_PACKS);
	//clock_t* times = (clock_t*) malloc (sizeof(clock_t)*NUM_AGENTS);		
	clock_t init, end;
	
	
	//Inicializar os vetores de Pacotes
	for(i=0 ; i<NUM_PACKS ; i++){
		packages[i] = rand() % 2;	//Valores entre 0 e 1
	}
	
	//Inicializar os Agentes e vetor de tempo
	for(i=0 ; i<NUM_AGENTS ; i++){
		agents[i] = rand() % NUM_PACKS; //Agentes Caem em valores aleatórios
		//times[i] = 0;					//Inicializando o vetor de tempos com zero(Não usado mais!)
	}
	
	//printVet(packages,NUM_PACKS);
	//printVet(agents,NUM_AGENTS);
	
	//Inicializando o CUDA
	if(!InitCUDA()) {
		return 0;
	}
	
	
	//Alocando os vetores no device
	cudaMalloc((void**)&d_packages, NUM_PACKS * sizeof(int));
	cudaMemcpy(d_packages, packages, NUM_PACKS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_agents, NUM_AGENTS * sizeof(int));
	cudaMemcpy(d_agents, agents, NUM_AGENTS * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMalloc((void**)&d_times, NUM_AGENTS * sizeof(clock_t));
	//cudaMemcpy(d_times, times, NUM_AGENTS * sizeof(clock_t), cudaMemcpyHostToDevice);
	
	/*
	Chamada do Kernel
	Cada thread do kernel é indexada por um agente, por isso o <<1,NUM_AGENTS>>>
	*/
	init = clock();
	kernel<<<1,NUM_AGENTS>>>(d_agents, NUM_AGENTS, d_packages, NUM_PACKS/*, d_times*/);
	end = clock();
	cudaMemcpy(packages, d_packages, NUM_PACKS * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(times, d_times, NUM_AGENTS * sizeof(clock_t), cudaMemcpyDeviceToHost);
	
	//printVet(packages,NUM_PACKS);
	//printVet(agents,NUM_AGENTS);
	//printVetFloat(times,NUM_AGENTS);
	printf("Tempo alternativo = %lu\n",(end-init));
	
	
	//Desalocando
	free(agents);
	free(packages);
	
	return 0;
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

void printVet(int* vet, int tam){
	int i;
	printf("Imprimindo Vetor:\n");
	for(i=0 ; i<tam ; i++){
		printf("%d  ",vet[i]);
	}
	printf("\n");
}


/*
void printVetFloat(clock_t* vet, int tam){
	int i;
	printf("Imprimindo Vetor de Tempos:\n");
	for(i=0 ; i<tam ; i++){
		printf("%lu  ",vet[i]);
	}
	printf("\n");
}
*/

__global__ void kernel(void** agents, int num_agents, void** packages, int num_packs/*, void** times*/){
	//clock_t init,end;
	//clock_t total;
	//init = clock();
	
	int bye=0;
	while(1){
		allPackagesAreGone(packages,num_packs,&bye);
		if(bye) break;
		
		//Confere a posição atual
		int posAtual = (int)agents[threadIdx.x];
		int caixaPosAtual = (int)packages[posAtual];
		
		//Caso haja um packote, pega o pacote
		if(caixaPosAtual == 1){
			packages[posAtual] = (void*) 0 ;
		}
		//Contrário decide para onde vai
		else{
			int dir = (posAtual + 1) % num_packs;
			int esq = (posAtual - 1) % num_packs;
			if(esq == -1){
				esq = num_packs - 1;
			}
			int caixaDir = (int)packages[dir];
			int caixaEsq = (int)packages[esq];
			//confere a direita
			if(caixaDir == 1){
				agents[threadIdx.x] = (void*)dir;
			}
			//confere a esquerda
			else{
				if(caixaEsq == 1){
					agents[threadIdx.x] = (void*)esq;
				}
				//randomicamente decide para onde vai
				else{
					int randomPos = (threadIdx.x) % 1;
					if(randomPos == 1){
						agents[threadIdx.x] = (void*)dir;
					}
					else{
						agents[threadIdx.x] = (void*)esq;
					}
				}
				
			}
			
		}
		
		
	}//end while
	//calculando final do processo
	//end = clock();
	//total = (init - end);
	
	//__syncthreads();
	//times[threadIdx.x] = (void*)total;
	
	return;
}

__device__ void allPackagesAreGone(void** packages, int num_packs, int* bye){
	int i,test;
	//__syncthreads();
	test=1;
	for(i=0 ; i<num_packs ; i++){
		int pos = (int)packages[i];
		if(pos == 1){
			test=0;
			break;
		}
	}
	if(test) *bye=1;
	else *bye=0;
	
}