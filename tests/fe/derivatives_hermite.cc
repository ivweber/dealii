#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/mapping_q1.h>

#include <string>

#include "../tests.h"

#include "derivatives.h"

#define PRECISION 8

template <int dim>
void
print_hermite_endpoint_derivatives()
{
    MappingQ<dim> m(1);
    
    FE_Hermite<dim> herm0(0);
    plot_function_derivatives<dim>(m, herm0, "Hermite-0");
    
    FE_Hermite<dim> herm1(1);
    plot_function_derivatives<dim>(m, herm1, "Hermite-1");
    
    //Skip the following for dim 3 or greater
    if (dim < 3)
    {
        FE_Hermite<dim> herm2(2);
        plot_function_derivatives<dim>(m, herm2, "Hermite-2");
    }
    if (dim == 1)
    {
        FE_Hermite<dim> herm3(3);
        plot_function_derivatives<dim>(m, herm3, "Hermite-3");
        
        FE_Hermite<dim> herm4(4);
        plot_function_derivatives<dim>(m, herm4, "Hermite-4");
    }
}

int
main()
{
    std::ofstream logfile("output");
    
    deallog << std::setprecision(PRECISION) << std::fixed;
    deallog.attach(logfile);
    
    print_hermite_endpoint_derivatives<1>();
    print_hermite_endpoint_derivatives<2>();
    print_hermite_endpoint_derivatives<3>();
    
    return 0;
}
