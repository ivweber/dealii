#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>

#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

using namespace dealii;

/**
 * Test case for Hermite with a time-stepping method for u_tt=u_xx,
 * to check that the stability conditions have transferred
 * correctly. Uses a manufactured solution of a plane wave
 * entering the domain at x=0 and leaving at x=3.
 * 
 * To avoid derivative discontinuities, use a Hermite polynomial
 * to define the wavefront's profile.
 * 
 * Solution is chosen so that initial conditions are not uniformly zero
 * and the wave is currently entering the domain through the boundary. 
 * This allows both the use of boundary projection and initial projectiion
 * for both value and derivatives to be tested.
 * 
 * Solution uses y = x-t-0.5 as a local coordinate and is as follows:
 * (100/3)y^2 - (2000/27)y^3            y in [0  , 0.3]
 * -4 + 40y - 100y^2 + (2000/27)y^3     y in [0.3, 0.6]
 * 0                                    otherwise
 * The right-most tail of the wave is already in the domain at t=0, but
 * can be excluded by setting the initial time to less that -0.1.
 */

template <int dim>
class Solution : public Function<dim>
{    
private:
    double current_time = 0;
    
public:
    Solution() {};
    Solution(double initial_time) : current_time(initial_time) {};
    
    double inline 
    update_time(double time_step)
    {
        return this->current_time += time_step;
    }
    
    double inline
    get_time()
    {
        return this->current_time;
    }
    
    virtual double 
    value(const Point<dim> &p, const unsigned int component) const override
    {
        (void)component;
        double y = p(0) + 0.5 - this->current_time;
        y *= 10/3;
        double output = 0;
        int ind = std::floor(y);
        switch (ind)
        {
            case 0:
                output = y * y * (3 - 2 * y);
                break;
            case 1:
                output = -4 + y*(12 - y*(9 - 2*y));
                break;
            default:
                break;
        }
        return output;
    }
};
