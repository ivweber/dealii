/*
 * Test case for Hermite on an irregular 1D grid. FE_Hermite<1>(reg)
 * should be able to perfectly represent any polynomial function up
 * to degree 2*reg+1. If all basis functions are correctly scaled
 * according to element size, then projecting a polynomial of this
 * form onto the FE space will produce negligible pointwise errors.
 */

//Standard library files to be used
#include <sstream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <iterator>

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

#define manual_laplace 0

using namespace dealii;

//Define a function to project onto the domain [-1,1]
class solution : public Function<1>
{    
public:
    virtual double
    value(const Point<1> &p, unsigned int c=0) const override 
    {
        return p(c) * (1.0 + p(c) * (0.5 - p(c)));
    }
    
    std::string
    get_function_string()
    {
        return "X + 0.5 X^2 - X^3";
    }
};

class rhs_function : public Function<1>
{
public:
    virtual double
    value(const Point<1> &p, unsigned int c=0) const override
    {
        return - 1.0 + 6 * p(c);
    }
};

void
test_fe_on_domain(const unsigned int regularity)
{
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
            if (distance < 1e-6)
            {
                cell->set_refine_flag();
                //break;
            }
        }
        tr.execute_coarsening_and_refinement();
    }
    
    FE_Hermite<1> herm(regularity);
    dof.distribute_dofs(herm);
    
    MappingHermite<1> mapping;
    
    QGauss<1> quadr(2*regularity + 2);
    
    Vector<double> sol(dof.n_dofs());
    Vector<double> rhs(dof.n_dofs());
    
    solution sol_object;
    rhs_function rhs_object;
    
    AffineConstraints<double> constraints;
    constraints.close();
    
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp);
    SparsityPattern sp;
    sp.copy_from(dsp);
    
    SparseMatrix<double> stiffness_matrix;
    stiffness_matrix.reinit(sp);
#if !manual_laplace
    MatrixCreator::create_laplace_matrix(mapping, dof, quadr, stiffness_matrix);
#endif 
    
    FEValues<1> fe_herm(mapping, herm, quadr, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    std::vector<types::global_dof_index> local_to_global(herm.n_dofs_per_cell());
 
    for (const auto &cell : dof.active_cell_iterators())
    {
        fe_herm.reinit(cell);
        cell->get_dof_indices(local_to_global);
        for (const unsigned int i : fe_herm.dof_indices())
        {
#if manual_laplace
            for (const unsigned int j : fe_herm.dof_indices())
            {
                double laplace_temp = 0;
                for (const unsigned int q : fe_herm.quadrature_point_indices())
                    laplace_temp += fe_herm.shape_grad(i,q)
                                    * fe_herm.shape_grad(j,q)
                                    * fe_herm.JxW(q);
                stiffness_matrix(local_to_global[i], local_to_global[j]) += laplace_temp;
            }
#endif
            
            double rhs_temp = 0;
            for (const unsigned int q : fe_herm.quadrature_point_indices())
                rhs_temp += fe_herm.shape_value(i,q) 
                            * rhs_object.value(fe_herm.quadrature_point(q))
                            * fe_herm.JxW(q);
            rhs(local_to_global[i]) += rhs_temp;
        }
    }

    std::map<types::global_dof_index, double> bound_vals;
    std::map<types::boundary_id, const Function<1,double>*> bound_map;
    bound_map.emplace(std::make_pair(0U, &sol_object));
    bound_map.emplace(std::make_pair(1U, &sol_object));
    
    VectorTools::project_boundary_values(dof, bound_map, QGauss<0>(1), bound_vals);
    MatrixTools::apply_boundary_values(bound_vals, stiffness_matrix, sol, rhs);
    
    SolverControl solver_deets(50, 1e-11);
    SolverCG<> solver(solver_deets);
    
    solver.solve(stiffness_matrix, sol, rhs, PreconditionIdentity() );
   
    DataOut<1> data;
    data.attach_dof_handler(dof);
    data.add_data_vector(sol, "Solution");
    data.build_patches(mapping, 29, DataOut<1>::CurvedCellRegion::curved_inner_cells);
    char filename[15];
    sprintf(filename, "solution-%d.vtu", regularity);
    std::ofstream outpt(filename);
    data.write_vtu(outpt);
    outpt.close();
    
    double err_sq = 0;
    
    for (auto &cell : dof.active_cell_iterators())
    {
        fe_herm.reinit(cell);
        cell->get_dof_indices(local_to_global);
        for (const unsigned int q : fe_herm.quadrature_point_indices())
        {
            double sol_at_point = 0;
            for (const unsigned int i : fe_herm.dof_indices())
                sol_at_point += fe_herm.shape_value(i,q) * sol(local_to_global[i]);
            sol_at_point -= sol_object.value(fe_herm.quadrature_point(q));
            err_sq += sol_at_point * sol_at_point * fe_herm.JxW(q);
        }
    }
    
    err_sq = std::sqrt(err_sq);
    
    deallog << std::endl;
    char fname[50];
    sprintf(fname, "Cell-1d-Hermite-%d", regularity);
    deallog.push(fname);
    
    deallog << "Test polynomial:" << std::endl;
    deallog << sol_object.get_function_string() << std::endl;
    deallog << std::endl;
    
    deallog << "Grid cells:" << std::endl;
    for (const auto &cell : tr.active_cell_iterators())
    {
        deallog << "(\t" << cell->vertex(0) << ","
                << "\t" << cell->vertex(1) << "\t)" << std::endl;
    }
    deallog << std::endl;
    
    deallog << "Interpolation error:" << std::endl;
    deallog << err_sq << "\n\n" << std::endl;
    deallog.pop();
}

int main()
{
    std::ofstream logfile("output");
    deallog << std::setprecision(8) << std::fixed;
    deallog.attach(logfile);
    
    test_fe_on_domain(0);
    test_fe_on_domain(1);
    test_fe_on_domain(2);
    test_fe_on_domain(3);
    
    return 0;
}
