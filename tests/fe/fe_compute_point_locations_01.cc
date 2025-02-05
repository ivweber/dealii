// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Test Functions::FEFieldFunction::compute_point_locations

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/fe_field_function.h>

#include <iostream>

#include "../tests.h"


template <int dim>
void
test_compute_pt_loc(unsigned int n_points)
{
  deallog << "Testing for dim = " << dim << std::endl;
  deallog << "Testing on: " << n_points << " points." << std::endl;

  // Creating a grid in the square [0,1]x[0,1]
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(std::max(6 - dim, 2));

  // Creating the finite elements needed:
  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Creating the random points
  std::vector<Point<dim>> points;

  for (std::size_t i = 0; i < n_points; ++i)
    points.push_back(random_point<dim>());

  std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
  std::vector<std::vector<Point<dim>>>                        qpoints;
  std::vector<std::vector<unsigned int>>                      maps;

  // Creating a dummy vector/fe_field_function in order to use FEFieldFunction
  Vector<double>                  dummy;
  Functions::FEFieldFunction<dim> fe_function(
    dof_handler, dummy, StaticMappingQ1<dim, dim>::mapping);
  std::size_t n_cells =
    fe_function.compute_point_locations(points, cells, qpoints, maps);

  deallog << "Points found in " << n_cells << " cells" << std::endl;

  // testing if the transformation is correct:
  // For each cell check if the quadrature points in the i-th FE
  // are the same as maps[i]
  for (unsigned int i = 0; i < cells.size(); ++i)
    {
      auto &cell      = cells[i];
      auto &quad      = qpoints[i];
      auto &local_map = maps[i];

      // Given the qpoints of the current cell, compute the real points
      FEValues<dim> fev(fe, quad, update_quadrature_points);
      fev.reinit(cell);
      const auto &real_quad = fev.get_quadrature_points();

      for (unsigned int q = 0; q < real_quad.size(); ++q)
        {
          // Check if points are the same as real points
          if (real_quad[q].distance(points[local_map[q]]) > 1e-10)
            deallog << "Error on cell : " << cell << " at local point " << i
                    << ", corresponding to real point " << points[local_map[q]]
                    << ", that got transformed to " << real_quad[q]
                    << " instead." << std::endl;
        }
    }
  deallog << "Test finished" << std::endl;
}

int
main()
{
  initlog();

  deallog << "Deal.II compute_pt_loc:" << std::endl;
  test_compute_pt_loc<2>(100);
  test_compute_pt_loc<3>(200);
}
