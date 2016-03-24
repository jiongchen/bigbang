#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "optimization.h"

using namespace alglib;

int main(int argc, char **argv)
{
    //
    // This example demonstrates minimization of F(x0,x1) = x0^2 + x1^2 -6*x0 - 4*x1,
    // with quadratic term given by sparse matrix structure.
    //
    // Exact solution is [x0,x1] = [3,2]
    //
    // We provide algorithm with starting point, although in this case
    // (dense matrix, no constraints) it can work without such information.
    //
    // IMPORTANT: this solver minimizes  following  function:
    //     f(x) = 0.5*x'*A*x + b'*x.
    // Note that quadratic term has 0.5 before it. So if you want to minimize
    // quadratic function, you should rewrite it in such way that quadratic term
    // is multiplied by 0.5 too.
    //
    // For example, our function is f(x)=x0^2+x1^2+..., but we rewrite it as
    //     f(x) = 0.5*(2*x0^2+2*x1^2) + ....
    // and pass diag(2,2) as quadratic term - NOT diag(1,1)!
    //
    sparsematrix a;
    real_1d_array b = "[-6,-4]";
    real_1d_array x0 = "[0,1]";
    real_1d_array s = "[1,1]";
    real_1d_array x;
    minqpstate state;
    minqpreport rep;

    // initialize sparsematrix structure
    sparsecreate(2, 2, 0, a);
    sparseset(a, 0, 0, 2.0);
    sparseset(a, 1, 1, 2.0);

    // create solver, set quadratic/linear terms
    minqpcreate(2, state);
    minqpsetquadratictermsparse(state, a, true);
    minqpsetlinearterm(state, b);
    minqpsetstartingpoint(state, x0);

    // Set scale of the parameters.
    // It is strongly recommended that you set scale of your variables.
    // Knowing their scales is essential for evaluation of stopping criteria
    // and for preconditioning of the algorithm steps.
    // You can find more information on scaling at http://www.alglib.net/optimization/scaling.php
    minqpsetscale(state, s);

    // solve problem with BLEIC-based QP solver.
    // default stopping criteria are used.
    minqpsetalgobleic(state, 0.0, 0.0, 0.0, 0);
    minqpoptimize(state);
    minqpresults(state, x, rep);
    printf("%s\n", x.tostring(2).c_str()); // EXPECTED: [3,2]

    // try to solve problem with Cholesky-based QP solver...
    // Oops! It does not support sparse matrices, -5 returned as completion code!
    minqpsetalgocholesky(state);
    minqpoptimize(state);
    minqpresults(state, x, rep);
    printf("%d\n", int(rep.terminationtype)); // EXPECTED: -5
    return 0;
}
