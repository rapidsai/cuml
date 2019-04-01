template <typename T>
magma_int_t magma_d_spmv	(	double 	alpha,
magma_d_matrix 	A,
magma_d_matrix 	x,
double 	beta,
magma_d_matrix 	y,
magma_queue_t 	queue 
)	
template <>
inline magma_int_t magma_d_spmv	(	double 	alpha,
magma_d_matrix 	A,
magma_d_matrix 	x,
double 	beta,
magma_d_matrix 	y,
magma_queue_t 	queue 
)	
{
return magma_d_spmv	( alpha, A, x, beta, y, );
}

template <>
inline magma_int_t magma_d_spmv	(	double 	alpha,
magma_d_matrix 	A,
magma_d_matrix 	x,
double 	beta,
magma_d_matrix 	y,
magma_queue_t 	queue 
)	
{
return magma_d_spmv	( alpha, A, x, beta, y, );
}

