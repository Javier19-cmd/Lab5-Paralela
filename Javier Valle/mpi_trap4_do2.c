/**
 * Javier Valle 20159
 * Mario de León 19019
 * 
 */

/* File:     mpi_trap4.c
 * Purpose:  Use MPI to implement a parallel version of the trapezoidal 
 *           rule. This version uses collective communications and
 *           MPI derived datatypes to distribute the input data and 
 *           compute the global sum.
 *
 * Input:    The endpoints of the interval of integration and the number
 *           of trapezoids
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Compile:  mpicc -o mpi_trap4 mpi_trap4.c -lm
 * Run:      mpiexec -n <number of processes> ./mpi_trap4
 *
 * Algorithm:
 *    1. Each process calculates "its" interval of
 *        integration.
 *    2. Each process estimates the integral of f(x)
 *        over its interval using the trapezoidal rule.
 *    3a. Each process != 0 sends its integral to 0.
 *    3b. Process 0 sums the calculations received from
 *        the individual processes and prints the result.
 *
 * Note:  f(x) is all hardwired.
 *
 * IPP:   Section 3.5 (pp. 117 and ff.)
 */
#include <stdio.h>
#include <mpi.h>
#include <math.h> // Add the math library for the -lm option

/* Build a derived datatype for distributing the input data */
void Build_mpi_type(double* a_p, double* b_p, int* n_p, MPI_Datatype* input_mpi_t_p);

/* Get the input values */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p);

/* Calculate local integral */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len);

/* Function we're integrating */
double f(double x);

int main(void) {
   int my_rank, comm_sz, n, local_n;
   double a, b, h, local_a, local_b;
   double local_int, total_int;

   /* Let the system do what it needs to start up MPI */
   MPI_Init(NULL, NULL);

   /* Get my process rank */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   /* Find out how many processes are being used */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   Get_input(my_rank, comm_sz, &a, &b, &n);

   h = (b - a) / n;          /* h is the same for all processes */
   local_n = n / comm_sz;    /* So is the number of trapezoids  */

   /* Length of each process' interval of
    * integration = local_n * h. So my interval
    * starts at: */
   local_a = a + my_rank * local_n * h;
   local_b = local_a + local_n * h;
   local_int = Trap(local_a, local_b, local_n, h);

   /* Add up the integrals calculated by each process */
   MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   /* Print the result */
   if (my_rank == 0) {
      printf("With n = %d trapezoids, our estimate\n", n);
      printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
   }

   /* Shut down MPI */
   MPI_Finalize();

   return 0;
}

/* Build a derived datatype for distributing the input data */
void Build_mpi_type(double* a_p, double* b_p, int* n_p, MPI_Datatype* input_mpi_t_p) {
   int array_of_blocklengths[3] = {1, 1, 1};
   MPI_Aint array_of_displacements[3];
   MPI_Aint a_addr, b_addr, n_addr;
   MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

   MPI_Get_address(a_p, &a_addr);
   MPI_Get_address(b_p, &b_addr);
   MPI_Get_address(n_p, &n_addr);

   array_of_displacements[0] = 0;
   array_of_displacements[1] = b_addr - a_addr;
   array_of_displacements[2] = n_addr - a_addr;

   MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, input_mpi_t_p);
   MPI_Type_commit(input_mpi_t_p);
}

/* Get the input values */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p) {
   MPI_Datatype input_mpi_t;

   Build_mpi_type(a_p, b_p, n_p, &input_mpi_t);

   if (my_rank == 0) {
      printf("Enter a, b, and n\n");
      scanf("%lf %lf %d", a_p, b_p, n_p);
   }

   /* Now you can use the MPI derived type to replace
    * multiple messages with a single one */
   MPI_Bcast(a_p, 1, input_mpi_t, 0, MPI_COMM_WORLD);
   MPI_Bcast(b_p, 1, input_mpi_t, 0, MPI_COMM_WORLD);
   MPI_Bcast(n_p, 1, input_mpi_t, 0, MPI_COMM_WORLD);

   MPI_Type_free(&input_mpi_t);
}

/* Calculate local integral */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
   double estimate, x;
   int i;

   estimate = (f(left_endpt) + f(right_endpt)) / 2.0;
   for (i = 1; i <= trap_count - 1; i++) {
      x = left_endpt + i * base_len;
      estimate += f(x);
   }
   estimate = estimate * base_len;

   return estimate;
}

/* Function we're integrating */
double f(double x) {
   return x * x;
}
