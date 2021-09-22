/*
 * Test case for Hermite on an irregular 1D grid. FE_Hermite<1>(reg)
 * should be able to perfectly represent any polynomial function up
 * to degree 2*reg+1. If all basis functions are correctly scaled
 * according to element size, then projecting a polynomial of this
 * form onto the FE space will produce negligible pointwise errors.
 */

//Standard library files to be used
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdlib>

//General Deal.II libraries to be used
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>  //To aid debugging

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_hermite.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

using namespace dealii;

//Define a function to project onto the domain
class test_poly : public Function<1>
{
public:
    virtual double
    value(const Point<1> &p, unsigned int c=0) const override {
        return p(0);// * (1.0 + 0.5 * p(0) - p(0) * p(0));
    }
};

void
test_fe_on_domain(const unsigned int regularity)
{
    AssertThrow(regularity != 0, ExcInternalError());   //Projecting a quadratic with linear polynomials won't be exact
    
    Triangulation<1> tr;
    DoFHandler<1> dof(tr);
    
    double left = -1.0, right = 1.0;
    Point<1> left_point(left), right_point(right);
    GridGenerator::hyper_cube(tr, left, right);  
    
    //Refine the right-most cell three times to get the elements [-1,0],[0,0.5],[0.5,0.75],[0.75,1]
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (auto &cell : tr.active_cell_iterators())
        {
            const double distance = right_point.distance(cell->vertex(1));
            if (true || distance < 1e-6)
            {
                cell->set_refine_flag();
                break;
            }
        }
        tr.execute_coarsening_and_refinement();
    }
    
    FE_Hermite<1> herm(regularity);
    dof.distribute_dofs(herm);
    
    MappingHermite<1> mapping;
    
    QGauss<1> quadr(2*regularity + 2);
    Vector<double> solution(dof.n_dofs());
    test_poly rhs_func;
    
    AffineConstraints<double> constraints;
    constraints.close();
    
    FEValues<1> fe_herm(mapping, herm, quadr, update_values | update_JxW_values);
    std::vector<types::global_dof_index> local_to_global(herm.n_dofs_per_cell());
#define project_manually 1    
#if project_manually
    FullMatrix<double> mass_matrix(dof.n_dofs(), dof.n_dofs());
    Vector<double> rhs_vec(dof.n_dofs());
    for (const auto &cell : dof.active_cell_iterators())
    {
        fe_herm.reinit(cell);
        cell->get_dof_indices(local_to_global);
        for (const unsigned int q : fe_herm.quadrature_point_indices())
            for (const unsigned int i : fe_herm.dof_indices())
            {
                for (const unsigned int j : fe_herm.dof_indices())
                {
                    mass_matrix(local_to_global[i], local_to_global[j]) += fe_herm.shape_value(i,q)
                                                                            * fe_herm.shape_value(j,q)
                                                                            * fe_herm.JxW(q);
                }
                rhs_vec(local_to_global[i]) += fe_herm.shape_value(i,q)
                                                * rhs_func.value(quadr.point(q))
                                                * fe_herm.JxW(q);
            }
    }
    IterationNumberControl solver_control(100, 1e-9);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(mass_matrix, solution, rhs_vec, PreconditionIdentity());
#else
    VectorTools::project(mapping, dof, constraints, quadr, rhs_func, solution, false);
#endif
    
    DataOut<1> data_out;
    data_out.attach_dof_handler(dof);
    data_out.add_data_vector(solution, "hermite_solution");
 //   data_out.build_patches(mapping, 29, DataOut<1>::curved_inner_cells);
    data_out.build_patches(29);
    char filename[20];
    sprintf(filename, "solution-%d.vtu", regularity);
    std::ofstream output(filename);
    data_out.write_vtu(output);
    output.close();
    
    std::ofstream text_output("printed_values.txt");
    for (unsigned int i = 0; i < solution.size(); ++i)
        text_output << solution(i) << "\t";
    text_output << std::endl;
    text_output.close();
    
#ifdef error_calculator   
    double err_sq = 0;
    
    for (auto &cell : tr.active_cell_iterators())
    {
        fe_herm.reinit(cell);
        cell->get_dof_indices(local_to_global);
        for (const unsigned int q : fe_herm.quadrature_point_indices())
        {
            double sol_at_point = 0;
            for (const unsigned int i : fe_herm.dof_indices())
                sol_at_point += fe_herm.shape_value(i,q) * solution(local_to_global[i]);
            sol_at_point -= rhs_func.value(fe_herm.point(q));
            err_sq += sol_at_point * sol_at_point * fe_herm.JxW(q);
        }
    }
    
    err_sq = std::sqrt(err_sq);
#endif
}

int main()
{
    test_fe_on_domain(1);
    test_fe_on_domain(2);
    test_fe_on_domain(3);
    
    return 0;
}
