// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2018 by the deal.II authors
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



// integrating \vec x over the surface of the [-1,1] hypercube and
// hyperball in 2d and 3d should yield zero
//
// same as q_point_sum_1, but with a C1 mapping


#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_c1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"



template <int dim>
void
check(const Triangulation<dim> &tria)
{
  MappingC1<dim> mapping;

  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  QGauss<dim - 1> q_face(3);

  FEFaceValues<dim> fe_face_values(
    mapping, fe, q_face, update_quadrature_points | update_JxW_values);
  FESubfaceValues<dim> fe_subface_values(
    mapping, fe, q_face, update_quadrature_points | update_JxW_values);

  Point<dim> n1, n2;
  for (typename DoFHandler<dim>::active_cell_iterator cell =
         dof_handler.begin_active();
       cell != dof_handler.end();
       ++cell)
    {
      // first integrate over faces
      // and make sure that the
      // result of the integration is
      // close to zero
      for (const unsigned int f : GeometryInfo<dim>::face_indices())
        if (cell->at_boundary(f))
          {
            fe_face_values.reinit(cell, f);
            for (unsigned int q = 0; q < q_face.size(); ++q)
              n1 += fe_face_values.quadrature_point(q) * fe_face_values.JxW(q);
          }

      // now same for subface
      // integration
      for (const unsigned int f : GeometryInfo<dim>::face_indices())
        if (cell->at_boundary(f))
          for (unsigned int sf = 0;
               sf < GeometryInfo<dim>::max_children_per_face;
               ++sf)
            {
              fe_subface_values.reinit(cell, f, sf);
              for (unsigned int q = 0; q < q_face.size(); ++q)
                n2 += fe_subface_values.quadrature_point(q) *
                      fe_subface_values.JxW(q);
            }
    }

  Assert(n1 * n1 < 1e-24, ExcInternalError());
  deallog << " face integration is ok: " << std::sqrt(n1 * n1) << std::endl;
  Assert(n2 * n2 < 1e-24, ExcInternalError());
  deallog << " subface integration is ok: " << std::sqrt(n2 * n2) << std::endl;
}


int
main()
{
  initlog();

  {
    Triangulation<2> coarse_grid;
    GridGenerator::hyper_cube(coarse_grid, -1, 1);
    check(coarse_grid);
  }

  // the C1 mapping is not fully
  // implemented in 3d, unfortunately
  //   {
  //     Triangulation<3> coarse_grid;
  //     GridGenerator::hyper_cube (coarse_grid, -1, 1);
  //     check (coarse_grid);
  //   }


  {
    Triangulation<2> coarse_grid;
    GridGenerator::hyper_ball(coarse_grid);
    static const SphericalManifold<2> boundary;
    coarse_grid.set_manifold(0, boundary);
    check(coarse_grid);
  }

  // the C1 mapping is not fully
  // implemented in 3d, unfortunately
  //   {
  //     Triangulation<3> coarse_grid;
  //     GridGenerator::hyper_ball (coarse_grid);
  //     static const SphericalManifold<3> boundary;
  //     coarse_grid.set_manifold (0, boundary);
  //     check (coarse_grid);
  //   }
}
