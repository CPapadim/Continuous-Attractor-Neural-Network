#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>

#include "minIni.h"
#include "simpson.h"
#include "randomlib.h"
/* 
This code was written by Charalampos Papadimitriou and is 
based on a continuous attractor neural network for spatial working
memory as described in Compte et al. 2000.  You may freely use, 
alter, and redistribute this code as long as this notice
remains intact.  You may not sell this code or otherwise profit
from this code by putting it behind a paywall, or any other method.
Any written material resulting from use of this
code must cite Papadimitriou et al. 2015  

Please address any inquiries to papadimitriou.c@gmail.com

NOTE:  This network uses the GotoBLAS2 library.  You must compile
this library prior to compiling the network code
*/

#define MATRIX_IDX(n, i, j) j*n + i
#define MATRIX_ELEMENT(A, m, n, i, j) A[ MATRIX_IDX(m, i, j) ]


/*Parameter File*/
const char paramfile[] = "parameters.ini";

/********************************/
/*****Parameter Declarations*****/
/********************************/

/***General Parameters***/
int Nt;				//Number of total neurons
int Ne;				//Number of excitatory neurons
int Ni;				//Number of inhibitory neurons
float Ne_Fraction;		//Fraction of cells that are excitatory

int numSims;			//Number of simulations
int tmax;			//Simulation time
float h;			//Integration time step
int nsteps;			//Number of total integration steps

int c1;				//Stimulation 1 cell range(from)
int c2;				//Stimulation 1 cell range(to)
float stim1_amp;		//Stimulation 1 amplitude
int stim1_interval1;		//Start of stimulation 1 (milliseconds)	
int stim1_interval2;		//End of stimulation 1 (milliseconds)
int stim1_interval1_n;		//Start of stimulation 1 (time step number)
int stim1_interval2_n;		//End of stimulation 1 (time step number)

int d1;				//Stimulation 2 cell range(from)
int d2;				//Stimulation 2 cell range(to)
float stim2_amp;		//Stimulation 2 amplitude
int stim2_interval1;		//Start of stimulation 2 (milliseconds)
int stim2_interval2;		//End of stimulation 2 (milliseconds)
int stim2_interval1_n;		//Start of stimulation 2 (time step number)
int stim2_interval2_n;		//End of stimulation 2 (time step number)

int e1;				//Reset signal range(from)
int e2;				//Reset signal range(to)
float stimR_amp;			//Reset signal amplitutde
int stimR_interval1;		//Start of reset signal (milliseconds)
int stimR_interval2;		//End of reset signal (milliseconds)
int stimR_interval1_n;		//Start of reset signal (time step number)
int stimR_interval2_n;		//End of reset signal (time step number)

int Cext;			//Number of external excitatory connections
float Vl;			//Leak potential
float Vthr;			//Voltage threshold for firing
float V_E;			//Excitatory synapse reversal potential
float V_I;			//Inhibitory synapse reversal potential
float Vreset;			//After a neuron fires, V is reset to this
float v_ext;			//External firing rate in Hz (per neuron)
float Cmp;			//Pyramidal membrane capacitance
float Cmi;			//Interneuron membrane capacitance
float gmp;			//Pyramidal leak conductance
float gmi;			//Interneuron leak conductance
float tau_rp_pyr;		//Pyramidal refractory period
float tau_rp_int;		//Interneuron refractory period
int n_rp_pyr;			//Pyramidal refractory period in time steps
int n_rp_int;			//Interneuron refractory period in time steps
float tau_ampa;		//Ampa time constant
float tau_gaba;		//Gaba time constant
float tau_nmda_rise;		//NMDA rising time constant
float tau_nmda_decay;		//NMDA decay time constant
float alpha;	 		//NMDA-related constant
float Mg;			//NMDA-related constant (Mg conc in mM)
float B; 			//NMDA-related constant

/*Conductance Variables - Pyramidal Cells*/
float gAMPA_ext_pyr;		//Ampa external conductances
float gGABA_pyr;		//Gaba conductances
float gNMDA_pyr;		//NMDA conductances

/*Conductance Variables - Interneurons*/
float gAMPA_ext_int;		//Ampa external conductances
float gGABA_int;		//Gaba conductances
float gNMDA_int;		//NMDA conductances

	
/*Variables defining connectivity footprint*/
float J_plus;		
float J_minus;
float sigma_ee;

/*Variables for saving summary data*/
int msec_window;
int sum_grp;
int sum_grp_size;

/*Other Variable Declarations*/
double NIntegW;
static float *W_Array = NULL;
static float *gNMDA_W_Array = NULL;
static float *gGABA_W_Array = NULL;
static float *gAMPA_ext = NULL;
static int *n_rp = NULL;
static float *s_ext = NULL;
static float *x_rand = NULL;
static float *s_ampa_rec = NULL;
static float *s_gaba = NULL;
static float *x_nmda = NULL;
static float *s_nmda = NULL;
static float *sgGABA = NULL;
static float *sgNMDA = NULL;
static float *V = NULL;
static int *refractory = NULL;
static float *Isyn = NULL;
static float *I_ampa_ext = NULL;
static float *I_ampa_rec = NULL;
static float *I_gaba = NULL;
static float *I_nmda = NULL;
static float *Istim1=NULL;
static float *Istim2=NULL;
static float *IstimR=NULL;
static int *spike_ext = NULL;
static int *spike_cell = NULL;
static int *spike_sum = NULL;
static float *Cm = NULL;
static float *gm = NULL;

static float *tempVector = NULL;
static float *tempVector2 = NULL;
static float *tempMatrix = NULL;

static char *filename = NULL;

/*Set Parameters from paramfile*/
void init_params()
{
	Nt=ini_getf("Network","Nt",1280,paramfile);
	Ne_Fraction=ini_getf("Network","Ne_Fraction",0.8,paramfile);
	Ne=(int)round(Ne_Fraction*Nt);
	Ni=(int)round((1-Ne_Fraction)*Nt);
	v_ext=(ini_getf("Network","v_ext1",1.8,paramfile))*(ini_getf("Network","Cext",1000,paramfile));
	J_plus=ini_getf("Network","J_plus",1.62,paramfile);
	sigma_ee=ini_getf("Network","sigma_ee",14.4,paramfile);

	numSims=ini_getf("Simulation","numSims",1,paramfile);
	tmax=ini_getf("Simulation","tmax",4500,paramfile);
	h=ini_getf("Simulation","h",0.1,paramfile);
	nsteps=(int)round(tmax/h);

	c1=ini_getf("Simulation","c1",501,paramfile);
	c2=ini_getf("Simulation","c2",600,paramfile);
	stim1_amp=ini_getf("Simulation","stim1_amp",-0.2,paramfile);
	stim1_interval1=ini_getf("Simulation","stim1_interval1",250,paramfile);
	stim1_interval2=ini_getf("Simulation","stim1_interval2",500,paramfile);
	stim1_interval1_n=round(stim1_interval1/h);
	stim1_interval2_n=round(stim1_interval2/h);

	d1=ini_getf("Simulation","d1",301,paramfile);
	d2=ini_getf("Simulation","d2",400,paramfile);
	stim2_amp=ini_getf("Simulation","stim2_amp",-0.2,paramfile);
	stim2_interval1=ini_getf("Simulation","stim2_interval1",2750,paramfile);
	stim2_interval2=ini_getf("Simulation","stim2_interval2",3000,paramfile);
	stim2_interval1_n=round(stim2_interval1/h);
	stim2_interval2_n=round(stim2_interval2/h);
	
	e1=0;
	e2=Ne;
	stimR_amp=ini_getf("Simulation","stimR_amp",-0.2,paramfile);
	stimR_interval1=ini_getf("Simulation","stimR_interval1",1500,paramfile);
	stimR_interval2=ini_getf("Simulation","stimR_interval2",2000,paramfile);
	stimR_interval1_n=round(stimR_interval1/h);
	stimR_interval2_n=round(stimR_interval2/h);
	
	Vl=ini_getf("Cell Parameters","Vl",-70,paramfile);
	Vthr=ini_getf("Cell Parameters","Vthr",-50,paramfile);
	V_E=ini_getf("Cell Parameters","V_E",0,paramfile);
	V_I=ini_getf("Cell Parameters","V_I",-70,paramfile);
	Vreset=ini_getf("Cell Parameters","Vreset",-60,paramfile);
	
	tau_ampa=ini_getf("Cell Parameters","tau_ampa",2,paramfile);
	tau_gaba=ini_getf("Cell Parameters","tau_gaba",10,paramfile);
	tau_nmda_rise=ini_getf("Cell Parameters","tau_nmda_rise",2,paramfile);
	tau_nmda_decay=ini_getf("Cell Parameters","tau_nmda_decay",100,paramfile);
	alpha=ini_getf("Cell Parameters","alpha",0.5,paramfile);
	Mg=ini_getf("Cell Parameters","Mg",1,paramfile);
	B=ini_getf("Cell Parameters","B",0.062,paramfile);

	Cmp=ini_getf("Cell Parameters","Cmp",0.005,paramfile);
	gmp=ini_getf("Cell Parameters","gmp",0.025,paramfile);
	tau_rp_pyr=ini_getf("Cell Parameters","tau_rp_pyr",2,paramfile);
	n_rp_pyr=(int)round(tau_rp_pyr/h);

	Cmi=ini_getf("Cell Parameters","Cmi",0.200,paramfile);
	gmi=ini_getf("Cell Parameters","gmi",0.020,paramfile);
	tau_rp_int=ini_getf("Cell Parameters","tau_rp_int",1,paramfile);
	n_rp_int=(int)round(tau_rp_pyr/h);

	gAMPA_ext_pyr=ini_getf("Synapse Conductances","gAMPA_ext_pyr",0.00285,paramfile);
	gGABA_pyr=ini_getf("Synapse Conductances","gGABA_pyr",0.0034,paramfile);
	gNMDA_pyr=gGABA_pyr/(ini_getf("Synapse Conductances","GABA_NMDA_RATIO_pyr",4.93595445,paramfile));
	
	gAMPA_ext_int=ini_getf("Synapse Conductances","gAMPA_ext_int",0.00218,paramfile);
	gGABA_int=ini_getf("Synapse Conductances","gGABA_int",0.00260909091,paramfile);
	gNMDA_int=gGABA_int/(ini_getf("Synapse Conductances","GABA_NMDA_RATIO_int",4.8405,paramfile));
}


/*==================================================*/


/*Initialize Matrix with a single value for all elements*/
void init_float_matrix(float* A, int m, int n, float element)
{
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
		{
			MATRIX_ELEMENT(A, m, n, i, j) = element;
		}
	}
}


void init_int_matrix(int* A, int m, int n, int element)
{
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < m; i++)
		{
			MATRIX_ELEMENT(A, m, n, i, j) = element;
		}
	}
}


/*Set the elements of the matrix within the specified range (m1 to m2 and n1 to n2) to a value*/
void setRange_float_matrix(float* A, int m, int m1, int m2, int n, int n1, int n2, float element)
{
	for (int j=n1; j < n2; j++)
	{
		for(int i=m1; i < m2; i++)
		{
			MATRIX_ELEMENT(A, m, n, i, j) = element;
		}
	}
}

void setRange_int_matrix(int* A, int m, int m1, int m2, int n, int n1, int n2, int element)
{
	for (int j=n1; j < n2; j++)
	{
		for(int i=m1; i < m2; i++)
		{
			MATRIX_ELEMENT(A, m, n, i, j) = element;
		}
	}
}

/*Add a value to the elements of a matrix*/
void consAdd_matrix(float* A, int m, int m1, int m2, int n, int n1, int n2,  float cons, float* B, int k, int k1, int l, int l1)
{
	for (int j = n1; j < n2; j++)
	{
		for (int i = m1; i < m2; i++)
		{
			int Bj=j-n1+l1;
			int Bi=i-m1+k1;
			MATRIX_ELEMENT(B, k, l, Bi, Bj)=cons+MATRIX_ELEMENT(A, m, n, i, j);
		}
	}
}



/*Add two integer matrices or matrix parts together and output the result in a third matrix*/
void add_int_matrix(int* A, int m, int m1, int m2, int n, int n1, int n2, int* B, int k, int k1, int l, int l1, int* C, int o, int o1, int p, int p1)
{
	for (int j = n1; j < n2; j++)
	{
		for (int i = m1; i < m2; i++)
		{
			int Bj=j-n1+l1;
			int Bi=i-m1+k1;
			int Cj=j-n1+p1;
			int Ci=i-m1+o1;
			MATRIX_ELEMENT(C, o, p, Ci, Cj)=MATRIX_ELEMENT(A, m, n, i, j)+MATRIX_ELEMENT(B, k, l, Bi, Bj);
		}
	}
}

/*Multiply part of a matrix A with a scalar and write it to B*/
/*A is m x n and part to be multiplied is in range m1 to m2 and n1 to n2*/
/*B is k x l and should be large enough to accomodate the result*/
/*k1 and l1 are offsets used so that data can be placed in any part of B*/
void scalarMultiply_matrix(float* A, int m, int m1, int m2, int n, int n1, int n2, float scal, float* B, int k, int k1, int l, int l1)
{
	for (int j = n1; j < n2; j++)
	{
		for (int i = m1; i < m2; i++)
		{
	//		MATRIX_ELEMENT(B, k, l, i-m1+k1, j-n1+l1)=scal*MATRIX_ELEMENT(A, m, n, i, j);
			int Bj=j-n1+l1;
			int Bi=i-m1+k1;
			MATRIX_ELEMENT(B, k, l, Bi, Bj)=scal*MATRIX_ELEMENT(A, m, n, i, j);
		
		}
	}
}

/*Multiply two matrices (or matrix parts) of the same size element by element (e.g. A1*B1, A2*B2, ..., AN*BN) and return the result in C*/
void elementwiseMultiply_matrix(float* A, int m, int m1, int m2, int n, int n1, int n2, float* B, int k, int k1, int l, int l1, float* C, int o, int o1, int p, int p1)
{
	for (int j = n1; j < n2; j++)
	{
		for(int i = m1; i < m2; i++)
		{
			int Bj=j-n1+l1;
			int Bi=i-m1+k1;
			int Cj=j-n1+p1;
			int Ci=j-m1+o1;
			MATRIX_ELEMENT(C, o, p, Ci, Cj)=MATRIX_ELEMENT(A, m, n, i, j)*MATRIX_ELEMENT(B, k, l, Bi, Bj);
		}
	}
}

/*Divide two matrices, same method as elementwiseMultiply_matrix*/
void elementwiseDivide_matrix(float* A, int m, int m1, int m2, int n, int n1, int n2, float* B, int k, int k1, int l, int l1, float* C, int o, int o1, int p, int p1)
{
	for (int j = n1; j < n2; j++)
	{
		for(int i = m1; i < m2; i++)
		{
			int Bj=j-n1+l1;
			int Bi=i-m1+k1;
			int Cj=j-n1+p1;
			int Ci=j-m1+o1;
			MATRIX_ELEMENT(C, o, p, Ci, Cj)=MATRIX_ELEMENT(A, m, n, i, j)/MATRIX_ELEMENT(B, k, l, Bi, Bj);
		}
	}
}



/*Take the exponential of each array value*/
void exp_matrix(float* A, int m, int m1, int m2, int n, int n1, int n2, float* B, int k, int k1, int l, int l1)
{
	for (int j = n1; j < n2; j++)
	{
		for(int i = m1; i < m2; i++)
		{
			int Bj=j-n1+l1;
			int Bi=i-m1+k1;
			MATRIX_ELEMENT(B, k, l, Bi, Bj)=exp(MATRIX_ELEMENT(A, m, n, i, j));
		}
	}

}


/*Output the Matrix contents to the Screen*/
void print_matrix(const float* A, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%8.7f  ", MATRIX_ELEMENT(A, m, n, i, j));
		}
		printf("\n");					     }
}


/*Random seed based on processor cycles since boot, instead of time, better seed generation for concurrent simulations that run at the same time and therefore would have the same time seed.  Only works on x86 processors.*/
unsigned long long rdtsc_seed()
{
	unsigned int lo,hi;
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return ((unsigned long long) hi << 32) | lo;
}

/*Generate random number in uniform distribution on the range N to M which is to be fed into a BETTER random number generator (randomlib.c) as a seed*/
int uniform_rand(int N, int M)
{
	return (int)(M + (rand() * (1.0 / (RAND_MAX + 1.0))) * (N - M));
}

/*Generate Random filename string*/
void gen_random_filename(char *s, const int len){
	static const char alphanum[] = 
		"0123456789"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz";
	for (int i = 0; i < len-5; i++) {
		//s[i] = alphanum[rand() / (RAND_MAX / (sizeof(alphanum)-1))];
		s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
	}
	s[len-5]='.';
	s[len-4]='d';
	s[len-3]='a';
	s[len-2]='t';
	s[len-1]='\0';
}


/*Function to numerically integrate when determining J_minus*/
double W_func_integ(double theta_diff)
{
	return exp((-(pow(theta_diff,2))/(2*(pow(sigma_ee,2)))));
}


/*Function used to calculate weights.  Returns the weight between two neurons N1 and N2*/
float W_func(float N1, float N2)
{
	float theta_diff = 360*(N1 - N2)/Ne;
	if(theta_diff > 180){theta_diff -= 360;}
	if(theta_diff < -180) {theta_diff +=360;}
	return J_minus+(J_plus-J_minus)*exp((-(pow(theta_diff,2))/(2*(pow(sigma_ee,2)))));
}


/*Numerical integration of gating variables (s) using rk2 method*/
float rk2_dsdt(float s, float tau, float alpha, float x, float dt)
{
	float k1 = dt*((-s/tau)+alpha*x*(1-s));
	float k2 = dt*((-(s+(k1/2))/tau)+alpha*x*(1-(s+(k1/2))));
	float dsdt = s+k2;
	return dsdt;
}

/*Numerical integration of voltage using rk2 method*/
float rk2_dvdt(float V, float Cm, float Vl, float gm, float Isyn, float dt)
{
	float k1 = dt*(-((gm*(V-Vl))+Isyn))/Cm;
	float k2 = dt*(-((gm*((V+(k1/2))-Vl))+Isyn))/Cm;
	float dvdt = V+k2;
	return dvdt;
}

/***************************************************/
/*Short Term Memory Network Initialization Function*/
/***************************************************/
void swmnet_init()
{
	
	init_params();  //Initialize parameters form INI file

	gen_random_filename(filename,20);
	/*Variables used to average over data*/
	msec_window=1;
	sum_grp=-1; //Initialize to -1 since the first time step will add 1 to it immediately, making it 0 (arrays are zero indexed so need to start at 0)
	sum_grp_size=(int)round(msec_window/h);
	spike_sum = malloc((int)(Nt * sizeof(int) * (int)round(nsteps/sum_grp_size)));
	if(spike_sum == NULL){printf("Out of Memory for spike_sum");}

	init_float_matrix(spike_sum,Nt,(int)round(nsteps/sum_grp_size),0);
	
	/***CREATE AN ARRAY OF WEIGHTS - START***/
	double (*f)(double);
	f = &W_func_integ;
	NIntegW=simpson(f,0,360,1000000);
	J_minus=(360-J_plus*(float)NIntegW)/(360-(float)NIntegW);
	
	W_Array = malloc(Nt * Nt * sizeof(float));
	if(W_Array == NULL){printf("Out of Memory for W_Array");}
	
	init_float_matrix(W_Array,Nt,Nt,1);

	for(int i=0; i<Ne; i++)
	{
		for(int j=0; j<Ne; j++)
		{
			MATRIX_ELEMENT(W_Array, Nt, Nt, i, j)=W_func(i,j);
		}
	}
	/***CREATE AN ARRAY OF WEIGHTS - END***/

	

	/*DEFINE NMDA AND GABA CONDUCTANCE ARRAYS FOR PYR. AND INT. CELLS - START*/
	//gNMDA_W_Array and gGABA_W_Array are size Nt * Nt instead of Ne x Nt and Ni x Nt because BLAS has really fast symmetric matrix functions as compared to the general matrix functions.  However, values beyond Ne x Nt and Ni x Nt are never actually used for anything
	gNMDA_W_Array = malloc(Nt * Nt * sizeof(float));
	if(gNMDA_W_Array == NULL){printf("Out of Memory for gNMDA_W_Array");}
	gGABA_W_Array = malloc(Ni * Nt * sizeof(float));
	if(gGABA_W_Array == NULL){printf("Out of Memory for gGABA_W_Array");}

	scalarMultiply_matrix(W_Array,Nt,0,Ne,Nt,0,Ne,gNMDA_pyr,gNMDA_W_Array,Nt,0,Nt,0);
	scalarMultiply_matrix(W_Array,Nt,0,Ne,Nt,Ne,Nt,gNMDA_int,gNMDA_W_Array,Nt,0,Nt,Ne);
	scalarMultiply_matrix(W_Array,Nt,Ne,Nt,Nt,0,Ne,gNMDA_int,gNMDA_W_Array,Nt,Ne,Nt,0);

	scalarMultiply_matrix(W_Array,Nt,Ne,Nt,Nt,Ne,Nt,gGABA_int,gGABA_W_Array,Ni,0,Nt,Ne);
	scalarMultiply_matrix(W_Array,Nt,Ne,Nt,Nt,0,Ne,gGABA_pyr,gGABA_W_Array,Ni,0,Nt,0);
	
//	print_matrix(gNMDA_W_Array,Ne,Nt);
//	printf("\n");
//	print_matrix(gGABA_W_Array,Ni,Nt);
	
	
	/*DEFINE NMDA AND GABA CONDACTANCE ARRAYS FOR PYR. AND INT. CELLS - END*/

	

	/*POPULATE SIMULATION ARRAYS WITH INITIAL VALUES - START*/
	
	refractory=malloc(Nt*sizeof(int));
	if(refractory == NULL){printf("Out of Memory for refractory");}
	Cm=malloc(Nt*sizeof(float));
	if(Cm== NULL){printf("Out of Memory for Cm");}
	gm=malloc(Nt*sizeof(float));
	if(gm == NULL){printf("Out of Memory for gm");}
	n_rp=malloc(Nt*sizeof(int));
	if(n_rp == NULL){printf("Out of Memory for n_rp");}
	gAMPA_ext=malloc(Nt*sizeof(float));
	if(gAMPA_ext == NULL){printf("Out of Memory for gAMPA_ext");}
	s_ext=malloc(Nt*sizeof(float));
	if(s_ext == NULL){printf("Out of Memory for s_ext");}
	x_rand=malloc(Nt*sizeof(float));
	if(x_rand == NULL){printf("Out of Memory for x_rand");}
	s_ampa_rec=malloc(Ne*sizeof(float));
	if(s_ampa_rec == NULL){printf("Out of Memory for s_ampa_rec");}
	
	
	//s_gaba and s_nmda are size Nt instead of Ni and Ne because BLAS has really fast symmetric matrix functions as compared to the general matrix functions.  However, values beyond Ni and Ne are never actually used for anything
	s_gaba=malloc(Ni*sizeof(float));
	if(s_gaba == NULL){printf("Out of Memory for s_gaba");}
	s_nmda=malloc(Nt*sizeof(float));
	if(s_nmda == NULL){printf("Out of Memory for s_nmda");}
	
	x_nmda=malloc(Ne*sizeof(float));
	if(x_nmda == NULL){printf("Out of Memory for x_nmda");}
	
	
	//sgGABA and sgNMDA are size Nt instead of Ni and Ne because BLAS has really fast symmetric matrix functions as compared to the general matrix functions.  However, values beyond Ni and Ne are never actually used for anything
	sgGABA=malloc(Nt*sizeof(float));
	if(sgGABA == NULL){printf("Out of Memory for sgGABA");}
	sgNMDA=malloc(Nt*sizeof(float));
	if(sgNMDA == NULL){printf("Out of Memory for sgNMDA");}
	
	V=malloc(Nt*sizeof(float));
	if(V == NULL){printf("Out of Memory for V");}
	Isyn=malloc(Nt*sizeof(float));
	if(Isyn == NULL){printf("Out of Memory for Isyn");}
	I_ampa_ext=malloc(Nt*sizeof(float));
	if(I_ampa_ext == NULL){printf("Out of Memory for I_ampa_ext");}
	I_ampa_rec=malloc(Nt*sizeof(float));
	if(I_ampa_rec == NULL){printf("Out of Memory for I_ampa_rec");}
	I_gaba=malloc(Nt*sizeof(float));
	if(I_gaba == NULL){printf("Out of Memory for I_gaba");}
	I_nmda=malloc(Nt*sizeof(float));
	if(I_nmda == NULL){printf("Out of Memory for I_nmda");}
	Istim1=malloc(Nt*sizeof(float));
	if(Istim1 == NULL){printf("Out of Memory for Istim1");}
	Istim2=malloc(Nt*sizeof(float));
	if(Istim2 == NULL){printf("Out of Memory for Istim2");}
	IstimR=malloc(Nt*sizeof(float));
	if(IstimR == NULL){printf("Out of Memory for IstimR");}
	spike_ext=malloc(Nt*sizeof(int));
	if(spike_ext == NULL){printf("Out of Memory for spike_ext");}
	spike_cell=malloc(Nt*sizeof(int));
	if(spike_cell == NULL){printf("Out of Memory for spike_cell");}

	tempVector=malloc(Nt*sizeof(float));
	if(tempVector == NULL){printf("Out of Memory for tempVector");}
	tempVector2=malloc(Nt*sizeof(float));
	if(tempVector2 == NULL){printf("Out of Memory for tempVector2");}
	tempMatrix=malloc(Nt*Nt*sizeof(float));
	if(tempMatrix == NULL){printf("Out of Memory for tempMatrix");}
	

	init_int_matrix(refractory,Nt,1,0);
	init_float_matrix(Cm,Nt,1,0);
	init_float_matrix(gm,Nt,1,0);
	init_int_matrix(n_rp,Nt,1,0);
	init_float_matrix(gAMPA_ext,Nt,1,0);
	init_float_matrix(s_ext,Nt,1,0);
	init_float_matrix(s_ampa_rec,Ne,1,0);
	init_float_matrix(s_gaba,Ni,1,0);
	init_float_matrix(x_nmda,Ne,1,0);
	init_float_matrix(s_nmda,Nt,1,0);
	init_float_matrix(V,Nt,1,-70);
	init_float_matrix(Isyn,Nt,1,0);
	init_float_matrix(I_ampa_ext,Nt,1,0);
	init_float_matrix(I_gaba,Nt,1,0);
	init_float_matrix(I_nmda,Nt,1,0);
	init_float_matrix(Istim1,Nt,1,0);
	init_float_matrix(Istim2,Nt,1,0);
	init_float_matrix(IstimR,Nt,1,0);
	init_int_matrix(spike_cell,Nt,1,0);
	init_int_matrix(spike_ext,Nt,1,0);

	/*POPULATE SIMULATION ARRAYS WITH INITIAL VALUES - END*/

	/*Set stimulation currents*/
	setRange_float_matrix(Istim1,Nt,c1,c2,1,0,1,stim1_amp);
	setRange_float_matrix(Istim2,Nt,d1,d2,1,0,1,stim2_amp);
	setRange_float_matrix(IstimR,Nt,e1,e2,1,0,1,stimR_amp);

	setRange_float_matrix(Cm,Nt,0,Ne,1,0,1,Cmp);
	setRange_float_matrix(Cm,Nt,Ne,Nt,1,0,1,Cmi);
	setRange_float_matrix(gm,Nt,0,Ne,1,0,1,gmp);
	setRange_float_matrix(gm,Nt,Ne,Nt,1,0,1,gmi);
	setRange_float_matrix(gAMPA_ext,Nt,0,Ne,1,0,1,gAMPA_ext_pyr);
	setRange_float_matrix(gAMPA_ext,Nt,Ne,Nt,1,0,1,gAMPA_ext_int);
	
	setRange_int_matrix(n_rp,Nt,0,Ne,1,0,1,n_rp_pyr);
	setRange_int_matrix(n_rp,Nt,Ne,Nt,1,0,1,n_rp_int);
	

}


/****Free Memory after a simulation is completed****/
void swmnet_deinit()
{
	free(spike_sum);
	free(W_Array);
	free(gNMDA_W_Array);
	free(gGABA_W_Array);
	free(refractory);
	free(Cm);
	free(gm);
	free(n_rp);
	free(gAMPA_ext);
	free(s_ext);
	free(x_rand);
	free(s_ampa_rec);
	free(s_gaba);
	free(s_nmda);
	free(x_nmda);
	free(sgGABA);
	free(sgNMDA);
	free(V);
	free(Isyn);
	free(I_ampa_ext);
	free(I_ampa_rec);
	free(I_gaba);
	free(I_nmda);
	free(Istim1);
	free(Istim2);
	free(IstimR);
	free(spike_ext);
	free(spike_cell);
	free(tempVector);
	free(tempVector2);
	free(tempMatrix);
}

/***********************************************/
/*Short Term Memory Network Processing Function*/
/***********************************************/

void swmnet()
{
	
	
	for(int n=0; n < nsteps; n++)
	{
		
		//If the cell is in refractory, decrease the refractory time by one time step
		for(int r=0; r < Nt; r++)
		{
			if(refractory[r] > 0){refractory[r] -= 1;}
		}
		
		
		/*Calculate gWeight*s for GABA and NMDA, which is equilvalent to multiplying each element and summing*/
		float dOne = 1.0;
		float dZero = 0.0;
		int iOne = 1;
		sgemv_("T",&Ni,&Nt,&dOne,gGABA_W_Array,&Ni,s_gaba,&iOne,&dZero,sgGABA,&iOne);
//		sgemv_("T",&Ne,&Nt,&dOne,gNMDA_W_Array,&Ne,s_nmda,&iOne,&dZero,sgNMDA,&iOne);
		
//		ssymv_("U",&Nt,&dOne,gGABA_W_Array,&Nt,s_gaba,&iOne,&dZero,sgGABA,&iOne);
		ssymv_("U",&Nt,&dOne,gNMDA_W_Array,&Nt,s_nmda,&iOne,&dZero,sgNMDA,&iOne);
		
	//	print_matrix(sgNMDA,Nt,1);
	//	printf("\n");
		for (int i = 0; i < Nt; i++)
		{
			
			/********************/
			/*Calculate Currents*/
			/********************/
			I_ampa_ext[i]=gAMPA_ext[i]*(V[i]-V_E)*s_ext[i];
			I_nmda[i]=((V[i]-V_E)/(1+exp(-B*V[i])/(Mg*3.57)))*sgNMDA[i];
			I_gaba[i]=(V[i]-V_I)*sgGABA[i];

			Isyn[i]=I_nmda[i]+I_ampa_ext[i]+I_gaba[i];
			if(n > stim1_interval1_n && n < stim1_interval2_n){Isyn[i]=Isyn[i]+Istim1[i];}
			if(n > stim2_interval1_n && n < stim2_interval2_n){Isyn[i]=Isyn[i]+Istim2[i];}
			if(n > stimR_interval1_n && n < stimR_interval2_n){Isyn[i]=Isyn[i]+IstimR[i];}
			
			spike_ext[i]=0;
			x_rand[i]=RandomDouble(0,1);
			if(x_rand[i] < h*v_ext*(pow(10,-3))){spike_ext[i]=1;}
			


			/************************/
			/*Calculate Currents End*/
			/************************/
		
		
			//Integrate s using rk2 method and then add a spike if one was fired
			s_ext[i]=rk2_dsdt(s_ext[i],tau_ampa,0,0,h)+spike_ext[i];
			
			if (i < Ne){
				x_nmda[i]=rk2_dsdt(x_nmda[i],tau_nmda_rise,0,0,h)+spike_cell[i];
				s_nmda[i]=rk2_dsdt(s_nmda[i],tau_nmda_decay,alpha,x_nmda[i],h);
			}
			if (i < Ni)
			{
				s_gaba[i]=rk2_dsdt(s_gaba[i],tau_gaba,0,0,h)+spike_cell[Ne+i];
			}
			
			if(refractory[i]<1)
			{
				V[i]=rk2_dvdt(V[i],Cm[i],Vl,gm[i],Isyn[i],h);
			}
			
			spike_cell[i]=0;
			if(V[i] > Vthr)
			{
				refractory[i]=(int)round(refractory[i])+((int)round(n_rp[i]+1));
				spike_cell[i]=1;
				V[i]=Vreset;
			}
			
		}
		

		//debug code start
//		int sum=0;
//		int sum_ext=0;
//		for (int i = 0; i < Nt; i++)
//		{
//			sum = sum + (int)spike_cell[i];
//
//			sum_ext=sum_ext+spike_ext[i];
//		}
//		printf("sum: %d\n",sum);
//		printf("sum_ext: %d\n",sum_ext);
//		printf("Neuron 300 V: %f\n",V[300]);

		//debug code end		
		if((n % sum_grp_size) == 0){
			sum_grp=sum_grp+1;
		}
	//	add_int_matrix(spike_sum,Nt,0,Nt,(int)round(nsteps/sum_grp_size),sum_grp,sum_grp+1,spike_cell,Nt,0,1,0,spike_sum,Nt,0,(int)round(nsteps/sum_grp_size),sum_grp);
		add_int_matrix(spike_sum,Nt,0,Nt,(int)round(nsteps/sum_grp_size),sum_grp,sum_grp+1,spike_cell,Nt,0,1,0,spike_sum,Nt,0,(int)round(nsteps/sum_grp_size),sum_grp);
		
	} //END n Loop
	FILE *datafile = fopen(filename,"w");
	fwrite(spike_sum, sizeof(spike_sum[0]), Nt*(nsteps/sum_grp_size), datafile); //Save in binary format (less space when integers are large and MATLAB can read this)

//SAVE IN TEXT FORMAT//	
//	for (int i0=0; i0 < Nt*(nsteps/sum_grp_size); i0++)
//	{
//		fprintf(datafile, "%d", spike_sum[i0]); //Save in text format (less space when integers are small)
//	}

	fclose(datafile);
}


	
/*****************/
/*Main C Function*/
/*****************/
int main(int argc, char *argv[])
{
	filename = malloc(sizeof(char)*20);
	if(filename == NULL){printf("Out of Memory for filename");}
	
	srand(rdtsc_seed());
	RandomInitialise(uniform_rand(1,29999),uniform_rand(1,29999)); //Seed the random number generator in randomlib.c with some pseudorandom seed generated by the simpler rand() algorithm
	
	numSims=ini_getf("Simulation","numSims",1,paramfile);

	for (int sim=1; sim <= numSims; sim++) {
		swmnet_init();
		swmnet();
		swmnet_deinit();
	}
	
	free(filename);

	/*SAMPLE USE OF GotoBLAS2 functions (dgemm in this case).  dgemm_ is the fortran interface and cblas_dgemm is the C interface.  Both can be used with no differences*/

/*	dgemm_(&transa,&transb,&size,&size,&size,&alpha,*a,&size,*b,&size,&beta,*c,&size);
	cblas_dgemm(&transa,&transb,&size,&size,&size,&alpha,*a,&size,*b,&size,&beta,*c,&size);*/

    return 0;
}
