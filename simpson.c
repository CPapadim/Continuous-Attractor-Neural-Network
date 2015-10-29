/*==============================================================
 *     Numerical integration of f(x) on [a,b]
 *     method: Simpson rule
 *     written by: Alex Godunov (February 2007)
 *---------------------------------------------------------------
 *     input:
 *        f   - a single argument real function (supplied by the user)
 *        a,b - the two end-points of the interval of integration
 *        n   - number of intervals
 *     output:
 *        s - result of integration
 *===============================================================*/

double simpson(double(*f)(double), double a, double b, int n)
{
	double s, dx, x;
	    // if n is odd - add +1 interval to make it even
		if((n/2)*2 != n) {n=n+1;}
	    	s = 0.0;
	    	dx = (b-a)/(float)n;
		int i;
	    	for (i=2; i<=n-1; i=i+2)
	    	{
	    		x = a+(float)i*dx;
	    		s = s + 2.0*f(x) + 4.0*f(x+dx);
	    	}
	    	s = (s + f(a)+f(b)+4.0*f(a+dx) )*dx/3.0;
	   	return s;
}
